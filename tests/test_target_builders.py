from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import LineString, Point, box

from omniwatermask.target_builders import (
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
        assert -90 <= bounds[1] <= 90   # min lat
        assert -90 <= bounds[3] <= 90   # max lat

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
    def test_builds_osm_water_targets(
        self, mock_osm, sample_rasterio_src, cache_dir
    ):
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
        )
        assert result is not None
        assert result.shape == (100, 100)
