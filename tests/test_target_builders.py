import logging
from queue import Queue
from unittest.mock import patch

import geopandas as gpd
import pytest
import torch
from shapely.geometry import LineString, Point, box

from omniwatermask.target_builders import (
    TargetBuildError,
    build_targets,
    combine_vector_targets,
    get_aux_data,
    get_osm_features,
    get_wgs84_bounds_gdf_from_raster,
)


class TestGetWgs84BoundsGdfFromRaster:
    def test_returns_geodataframe_in_4326(self, sample_rasterio_src):
        result = get_wgs84_bounds_gdf_from_raster(sample_rasterio_src)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326

    def test_bounds_are_valid(self, sample_rasterio_src):
        result = get_wgs84_bounds_gdf_from_raster(sample_rasterio_src)
        bounds = result.total_bounds
        # Should produce valid WGS84 coordinates
        assert -180 <= bounds[0] <= 180  # min lon
        assert -180 <= bounds[2] <= 180  # max lon
        assert -90 <= bounds[1] <= 90  # min lat
        assert -90 <= bounds[3] <= 90  # max lat

    def test_single_geometry(self, sample_rasterio_src):
        result = get_wgs84_bounds_gdf_from_raster(sample_rasterio_src)
        assert len(result) == 1


class TestGetOsmFeatures:
    @patch("omniwatermask.target_builders.ox.features_from_bbox")
    def test_returns_geodataframe(self, mock_features, sample_geodataframe_4326):
        mock_gdf = gpd.GeoDataFrame(
            geometry=[box(115.85, -31.96, 115.87, -31.94)],
            crs="EPSG:4326",
        )
        mock_features.return_value = mock_gdf
        result = get_osm_features(sample_geodataframe_4326, tags={"natural": "water"})
        assert isinstance(result, gpd.GeoDataFrame)

    @patch("omniwatermask.target_builders.ox.features_from_bbox")
    def test_handles_insufficient_response(
        self, mock_features, sample_geodataframe_4326
    ):
        from osmnx._errors import InsufficientResponseError

        mock_features.side_effect = InsufficientResponseError("No data")
        result = get_osm_features(sample_geodataframe_4326, tags={"natural": "water"})
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    @patch("omniwatermask.target_builders.ox.features_from_bbox")
    def test_propagates_an_unparsable_response(
        self, mock_features, sample_geodataframe_4326
    ):
        """osmnx reports a 200 whose body is not JSON as InsufficientResponseError
        too, chained from the JSONDecodeError. Swallowing that would cache a
        fetch failure as "no features here"."""
        from json import JSONDecodeError

        from osmnx._errors import InsufficientResponseError

        error = InsufficientResponseError("Overpass responded: 200 OK <html>")
        error.__cause__ = JSONDecodeError("Expecting value", "<html>", 0)
        mock_features.side_effect = error

        with pytest.raises(InsufficientResponseError):
            get_osm_features(sample_geodataframe_4326, tags={"natural": "water"})


class TestOsmnxLogging:
    """osmnx pauses for Overpass rate limiting and retries 429/504 internally.

    It reports both through its own log, which by default goes nowhere - so the
    wait looks like a hang.
    """

    @patch("omniwatermask.target_builders.ox.features_from_bbox")
    def test_osmnx_messages_reach_standard_logging(
        self, mock_features, sample_geodataframe_4326, caplog
    ):
        import osmnx as ox

        mock_features.return_value = gpd.GeoDataFrame(
            geometry=[box(115.85, -31.96, 115.87, -31.94)], crs="EPSG:4326"
        )
        get_osm_features(sample_geodataframe_4326, tags={"natural": "water"})

        with caplog.at_level(logging.WARNING, logger=ox.settings.log_name):
            ox.utils.log("responded 429: we'll retry in 55 secs", level=logging.WARNING)
        assert "we'll retry in 55 secs" in caplog.text

    @patch("omniwatermask.target_builders.ox.features_from_bbox")
    def test_does_not_write_an_osmnx_log_file(
        self, mock_features, sample_geodataframe_4326, tmp_path
    ):
        """Routing the messages must not turn on osmnx's own file logging."""
        import osmnx as ox

        mock_features.return_value = gpd.GeoDataFrame(
            geometry=[box(115.85, -31.96, 115.87, -31.94)], crs="EPSG:4326"
        )
        logs_folder = tmp_path / "logs"
        with patch.object(ox.settings, "logs_folder", str(logs_folder)):
            get_osm_features(sample_geodataframe_4326, tags={"natural": "water"})
            ox.utils.log("a message", level=logging.WARNING)
        assert not logs_folder.exists()


class TestGetAuxData:
    def test_reads_vector_file(self, tmp_path, sample_geodataframe_4326):
        vector_path = tmp_path / "aux.geojson"
        gdf = gpd.GeoDataFrame(
            geometry=[box(115.85, -31.96, 115.87, -31.94)],
            crs="EPSG:4326",
        )
        gdf.to_file(vector_path, driver="GeoJSON")

        result = get_aux_data(bbox=sample_geodataframe_4326, vector_path=vector_path)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326


class TestCombineVectorTargets:
    def test_combines_polygons(self, sample_rasterio_src):
        gdf1 = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390500, 6460500)], crs="EPSG:32650"
        )
        gdf2 = gpd.GeoDataFrame(
            geometry=[box(390500, 6460500, 390800, 6460800)], crs="EPSG:32650"
        )
        result = combine_vector_targets([gdf1, gdf2], sample_rasterio_src)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2

    def test_returns_none_for_all_empty(self, sample_rasterio_src):
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32650")
        result = combine_vector_targets([empty], sample_rasterio_src)
        assert result is None

    def test_filters_points(self, sample_rasterio_src):
        gdf = gpd.GeoDataFrame(
            geometry=[Point(390500, 6460500), box(390200, 6460200, 390500, 6460500)],
            crs="EPSG:32650",
        )
        result = combine_vector_targets([gdf], sample_rasterio_src)
        # Point should be filtered out, only polygon remains
        assert len(result) == 1

    def test_buffers_lines(self, sample_rasterio_src):
        line = LineString([(390200, 6460200), (390800, 6460800)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:32650")
        result = combine_vector_targets([gdf], sample_rasterio_src)
        # Line should be buffered into a polygon
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "Polygon"


class TestBuildTargets:
    def test_returns_none_when_no_targets_enabled(self, sample_rasterio_src, cache_dir):
        result = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=False,
            osm_roads=False,
            osm_buildings=False,
        )
        assert result is None

    def test_returns_none_via_queue(self, sample_rasterio_src, cache_dir):
        from queue import Queue

        q = Queue()
        result = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=False,
            osm_roads=False,
            osm_buildings=False,
            queue=q,
        )
        assert result is q
        assert q.get() is None

    @patch("omniwatermask.target_builders.get_osm_features")
    def test_builds_osm_water_targets(self, mock_osm, sample_rasterio_src, cache_dir):
        from omniwatermask.vector_cache import initialize_db

        initialize_db(cache_dir)
        water_poly = box(390200, 6460200, 390800, 6460800)
        mock_osm.return_value = gpd.GeoDataFrame(
            geometry=[water_poly], crs="EPSG:32650"
        )
        result = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            osm_roads=False,
            osm_buildings=False,
            use_cache=False,
            vector_source="osm",
        )
        assert result is not None
        assert result.shape == (100, 100)

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_builds_overture_water_targets_by_default(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        from omniwatermask.vector_cache import initialize_db

        initialize_db(cache_dir)
        water_poly = box(390200, 6460200, 390800, 6460800)
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[water_poly], crs="EPSG:32650"
        )
        result = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            osm_roads=False,
            osm_buildings=False,
            use_cache=False,
        )
        assert result is not None
        assert result.shape == (100, 100)
        assert mock_overture.call_args.kwargs["kind"] == "water"

    @patch("omniwatermask.target_builders.get_overture_features")
    @patch("omniwatermask.target_builders.get_osm_features")
    def test_osm_source_does_not_call_overture(
        self, mock_osm, mock_overture, sample_rasterio_src, cache_dir
    ):
        mock_osm.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            use_cache=False,
            vector_source="osm",
        )
        assert mock_osm.called
        assert not mock_overture.called

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_maps_each_kind_to_its_fetcher(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            osm_roads=True,
            osm_buildings=True,
            use_cache=False,
        )
        kinds = [call.kwargs["kind"] for call in mock_overture.call_args_list]
        assert kinds == ["water", "roads", "buildings"]

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_ocean_flag_forwarded_to_overture(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            use_cache=False,
            include_ocean=False,
        )
        assert mock_overture.call_args.kwargs["include_ocean"] is False

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_negative_targets_never_request_ocean(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        """Ocean only affects water, so building/road builds record it as off.

        Otherwise toggling include_ocean would invalidate cached negative
        targets it cannot change.
        """
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=False,
            osm_roads=True,
            osm_buildings=True,
            use_cache=False,
            include_ocean=True,
        )
        for call in mock_overture.call_args_list:
            assert call.kwargs["include_ocean"] is False


class TestBuildTargetsErrorHandling:
    """A failed build must be distinguishable from an empty one.

    Both used to come back as None, so the pipeline exported a mask built
    without its vector targets and could not tell that anything had gone wrong.
    """

    @staticmethod
    def _build(raster_src, cache_dir, **kwargs):
        return build_targets(
            raster_src=raster_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            use_cache=False,
            **kwargs,
        )

    def test_unknown_vector_source_raises(self, sample_rasterio_src, cache_dir):
        with pytest.raises(ValueError, match="Unknown vector_source"):
            self._build(sample_rasterio_src, cache_dir, vector_source="overtrue")

    def test_unknown_vector_source_raises_even_with_a_queue(
        self, sample_rasterio_src, cache_dir
    ):
        """The threaded call path must not convert a config error into a None."""
        q = Queue()
        with pytest.raises(ValueError, match="Unknown vector_source"):
            self._build(
                sample_rasterio_src, cache_dir, vector_source="overtrue", queue=q
            )
        assert q.empty()

    @patch("omniwatermask.target_builders.ox")
    def test_outdated_osmnx_raises(self, mock_ox, sample_rasterio_src, cache_dir):
        mock_ox.__version__ = "1.9.0"
        with pytest.raises(ImportError, match="too old"):
            self._build(sample_rasterio_src, cache_dir, vector_source="osm")

    @patch("omniwatermask.target_builders.get_overture_features")
    @patch("omniwatermask.target_builders.ox")
    def test_outdated_osmnx_does_not_block_the_overture_path(
        self, mock_ox, mock_overture, sample_rasterio_src, cache_dir
    ):
        """osmnx is only used for vector_source="osm"; requiring 2.0 of it to
        read Overture parquet would strand users on an old install."""
        mock_ox.__version__ = "1.9.0"
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        assert self._build(sample_rasterio_src, cache_dir) is not None

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_fetch_failure_raises(self, mock_overture, sample_rasterio_src, cache_dir):
        mock_overture.side_effect = RuntimeError("Overture water fetch failed")
        with pytest.raises(TargetBuildError) as excinfo:
            self._build(sample_rasterio_src, cache_dir)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_fetch_failure_is_queued_rather_than_raised(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        """This runs as a thread target, where a raise would be lost."""
        q = Queue()
        mock_overture.side_effect = RuntimeError("Overture water fetch failed")
        self._build(sample_rasterio_src, cache_dir, queue=q)
        queued = q.get()
        assert isinstance(queued, TargetBuildError)
        assert isinstance(queued.__cause__, RuntimeError)

    def test_nothing_to_build_is_not_a_failure(self, sample_rasterio_src, cache_dir):
        """None means "no targets were asked for" and must stay distinct from a
        failure, or every vector-free run would look like an error."""
        result = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=False,
            osm_roads=False,
            osm_buildings=False,
        )
        assert result is None

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_fetch_failure_is_logged(
        self, mock_overture, sample_rasterio_src, cache_dir, caplog
    ):
        """The traceback is what makes a skipped scene diagnosable."""
        mock_overture.side_effect = RuntimeError("Overture water fetch failed")
        with caplog.at_level(logging.ERROR):
            with pytest.raises(TargetBuildError):
                self._build(sample_rasterio_src, cache_dir)
        assert "Overture water fetch failed" in caplog.text

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_fetch_failure_is_not_cached(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        """A cached failure would make one bad network moment permanent."""
        from omniwatermask.vector_cache import initialize_db

        initialize_db(cache_dir)
        mock_overture.side_effect = [
            RuntimeError("Overture water fetch failed"),
            gpd.GeoDataFrame(
                geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
            ),
        ]
        with pytest.raises(TargetBuildError):
            build_targets(
                raster_src=sample_rasterio_src,
                aux_vector_sources=[],
                device="cpu",
                cache_dir=cache_dir,
                osm_water=True,
                use_cache=True,
            )
        second = build_targets(
            raster_src=sample_rasterio_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            use_cache=True,
        )
        assert second is not None, "the retry must refetch, not read back the failure"


class TestBuildTargetsVectorCache:
    """The v2 cache keys on source and ocean.

    Without those columns a cached Overture build would be served for an OSM
    request (and vice versa), silently returning the wrong vectors.
    """

    @staticmethod
    def _build(raster_src, cache_dir, **kwargs):
        from omniwatermask.vector_cache import initialize_db

        initialize_db(cache_dir)
        return build_targets(
            raster_src=raster_src,
            aux_vector_sources=[],
            device="cpu",
            cache_dir=cache_dir,
            osm_water=True,
            use_cache=True,
            **kwargs,
        )

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_second_identical_build_hits_cache(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        first = self._build(sample_rasterio_src, cache_dir)
        second = self._build(sample_rasterio_src, cache_dir)

        assert first is not None and second is not None
        assert mock_overture.call_count == 1, "second build should not refetch"
        assert torch.equal(first, second)

    @patch("omniwatermask.target_builders.get_osm_features")
    @patch("omniwatermask.target_builders.get_overture_features")
    def test_osm_build_does_not_reuse_overture_cache(
        self, mock_overture, mock_osm, sample_rasterio_src, cache_dir
    ):
        poly = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        mock_overture.return_value = poly
        mock_osm.return_value = poly

        self._build(sample_rasterio_src, cache_dir, vector_source="overture")
        self._build(sample_rasterio_src, cache_dir, vector_source="osm")

        assert mock_overture.call_count == 1
        assert mock_osm.call_count == 1, "osm build must not reuse the overture entry"

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_ocean_flag_change_invalidates_water_cache(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        self._build(sample_rasterio_src, cache_dir, include_ocean=True)
        self._build(sample_rasterio_src, cache_dir, include_ocean=False)

        assert mock_overture.call_count == 2

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_ocean_flag_change_does_not_invalidate_negative_targets(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        """Ocean cannot change road/building targets, so flipping it must not
        force them to be refetched."""
        from omniwatermask.vector_cache import initialize_db

        initialize_db(cache_dir)
        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        for include_ocean in (True, False):
            build_targets(
                raster_src=sample_rasterio_src,
                aux_vector_sources=[],
                device="cpu",
                cache_dir=cache_dir,
                osm_water=False,
                osm_roads=True,
                osm_buildings=True,
                use_cache=True,
                include_ocean=include_ocean,
            )
        # roads + buildings on the first build only; the second is a cache hit.
        assert mock_overture.call_count == 2

    @patch("omniwatermask.target_builders.get_overture_features")
    def test_records_source_and_ocean_in_cache_db(
        self, mock_overture, sample_rasterio_src, cache_dir
    ):
        from omniwatermask.vector_cache import view_cache_db

        mock_overture.return_value = gpd.GeoDataFrame(
            geometry=[box(390200, 6460200, 390800, 6460800)], crs="EPSG:32650"
        )
        self._build(sample_rasterio_src, cache_dir, include_ocean=True)
        df = view_cache_db(cache_dir)
        assert len(df) == 1
        assert df["source"].iloc[0] == "overture"
        assert bool(df["ocean"].iloc[0]) is True
