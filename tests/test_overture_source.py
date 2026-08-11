from unittest.mock import patch

import geopandas as gpd
import pyarrow as pa
import pytest
from shapely.geometry import box

from omniwatermask.overture_source import MAX_ATTEMPTS, get_overture_features


@pytest.fixture(autouse=True)
def no_backoff_sleep():
    """Keep the retry backoff from actually sleeping through the unit suite."""
    with patch("omniwatermask.overture_source.time.sleep") as mock_sleep:
        yield mock_sleep


@pytest.fixture(autouse=True)
def stac_reports_files_present():
    """Default the STAC coverage check to "files exist, so a None is an error".

    Left unpatched it would make a real network call, so every test that lets
    the reader return None must pin it.
    """
    with patch("omniwatermask.overture_source.core._prepare_query") as mock_prepare:
        mock_prepare.return_value = ("dataset", None)
        yield mock_prepare


def _overture_gdf(
    subtypes, bounds_gdf=None, geometries=None, classes=None, crs="EPSG:4326"
):
    """Build a GeoDataFrame shaped like an Overture response.

    Geometries default to the full request bounds so they survive the clip.
    """
    if geometries is None:
        assert bounds_gdf is not None, "pass bounds_gdf or explicit geometries"
        geometries = [box(*bounds_gdf.total_bounds)] * len(subtypes)
    return gpd.GeoDataFrame(
        {
            "id": [f"id_{i}" for i in range(len(subtypes))],
            "subtype": subtypes,
            "class": classes if classes is not None else ["unknown"] * len(subtypes),
            # Overture carries nested attribute columns that must not reach the
            # parquet cache.
            "sources": [[{"dataset": "OpenStreetMap"}]] * len(subtypes),
            "geometry": geometries,
        },
        crs=crs,
    )


class TestGetOvertureFeatures:
    def test_rejects_unknown_kind(self, sample_geodataframe_4326):
        with pytest.raises(ValueError, match="Unknown Overture feature kind"):
            get_overture_features(sample_geodataframe_4326, kind="coastline")

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_raises_when_reader_is_none(self, mock_reader, sample_geodataframe_4326):
        """record_batch_reader returns None on network error rather than raising."""
        mock_reader.return_value = None
        with pytest.raises(RuntimeError, match="failed after"):
            get_overture_features(sample_geodataframe_4326, kind="water")

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_returns_empty_for_no_features(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_reader.return_value = pa.RecordBatchReader
        mock_from_arrow.return_value = gpd.GeoDataFrame()
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_uses_stac(self, mock_reader, mock_from_arrow, sample_geodataframe_4326):
        """stac=True is ~5x faster; the client default is False."""
        mock_from_arrow.return_value = _overture_gdf(["lake"], sample_geodataframe_4326)
        get_overture_features(sample_geodataframe_4326, kind="water")
        assert mock_reader.call_args.kwargs["stac"] is True

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_requests_correct_type_per_kind(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(["road"], sample_geodataframe_4326)
        for kind, expected in [
            ("water", "water"),
            ("roads", "segment"),
            ("buildings", "building"),
        ]:
            get_overture_features(sample_geodataframe_4326, kind=kind)
            assert mock_reader.call_args.args[0] == expected

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_drops_nested_columns(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(["lake"], sample_geodataframe_4326)
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert set(result.columns) == {"subtype", "class", "geometry"}

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_excludes_ocean_when_disabled(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["ocean", "lake", "river"], sample_geodataframe_4326
        )
        result = get_overture_features(
            sample_geodataframe_4326, kind="water", include_ocean=False
        )
        assert "ocean" not in set(result["subtype"])
        assert len(result) == 2

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_includes_ocean_by_default(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["ocean", "lake"], sample_geodataframe_4326
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert "ocean" in set(result["subtype"])

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_keeps_only_road_segments(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """highway=* had no rail or waterway equivalent, so match that."""
        mock_from_arrow.return_value = _overture_gdf(
            ["road", "rail", "water", "road"], sample_geodataframe_4326
        )
        result = get_overture_features(sample_geodataframe_4326, kind="roads")
        assert set(result["subtype"]) == {"road"}
        assert len(result) == 2

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_returns_empty_when_filter_removes_everything(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["rail", "water"], sample_geodataframe_4326
        )
        result = get_overture_features(sample_geodataframe_4326, kind="roads")
        assert len(result) == 0

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_buildings_are_not_subtype_filtered(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["residential", "commercial", "outbuilding"], sample_geodataframe_4326
        )
        result = get_overture_features(sample_geodataframe_4326, kind="buildings")
        assert len(result) == 3

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_clips_to_bounds(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        bounds = sample_geodataframe_4326.total_bounds
        # A feature extending well beyond the eastern edge of the bounds.
        wide = box(bounds[0], bounds[1], bounds[2] + 1.0, bounds[3])
        mock_from_arrow.return_value = _overture_gdf(["lake"], geometries=[wide])
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert result.total_bounds[2] == pytest.approx(bounds[2])

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_passes_bbox_from_bounds(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(["lake"], sample_geodataframe_4326)
        get_overture_features(sample_geodataframe_4326, kind="water")
        bbox = mock_reader.call_args.kwargs["bbox"]
        assert bbox == pytest.approx(tuple(sample_geodataframe_4326.total_bounds))

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_surfaces_persistent_reader_errors(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """A cloud storage failure must surface rather than be swallowed into an
        empty frame, which the pipeline would read as "no water here"."""
        mock_reader.side_effect = OSError("connection reset")
        with pytest.raises(RuntimeError, match="failed after") as excinfo:
            get_overture_features(sample_geodataframe_4326, kind="water")
        assert isinstance(excinfo.value.__cause__, OSError)
        assert "connection reset" in str(excinfo.value.__cause__)

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_sets_crs_when_response_has_none(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """Overture geometries are always WGS84, but the arrow frame may not
        carry CRS metadata. Without set_crs the clip against the 4326 bounds
        would raise."""
        mock_from_arrow.return_value = _overture_gdf(
            ["lake"], sample_geodataframe_4326, crs=None
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert result.crs.to_epsg() == 4326
        assert len(result) == 1

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_handles_response_without_subtype_column(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """Guard the schema-drift path: no subtype means no subtype filtering,
        rather than a KeyError."""
        bounds = sample_geodataframe_4326.total_bounds
        mock_from_arrow.return_value = gpd.GeoDataFrame(
            {"id": ["a"], "geometry": [box(*bounds)]}, crs="EPSG:4326"
        )
        result = get_overture_features(sample_geodataframe_4326, kind="roads")
        assert len(result) == 1
        assert set(result.columns) == {"geometry"}

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_ocean_flag_does_not_affect_non_water_kinds(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["residential", "commercial"], sample_geodataframe_4326
        )
        result = get_overture_features(
            sample_geodataframe_4326, kind="buildings", include_ocean=False
        )
        assert len(result) == 2


class TestFetchRetry:
    """Overture is served from S3, where failures are usually transient."""

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_retries_until_a_reader_call_succeeds(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_reader.side_effect = [
            OSError("connection reset"),
            pa.RecordBatchReader,
        ]
        mock_from_arrow.return_value = _overture_gdf(["lake"], sample_geodataframe_4326)
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert len(result) == 1
        assert mock_reader.call_count == 2

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_retries_failures_raised_while_streaming(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """The reader streams lazily, so a failed range read surfaces out of
        from_arrow rather than out of the reader call."""
        mock_from_arrow.side_effect = [
            OSError("S3 SlowDown"),
            _overture_gdf(["lake"], sample_geodataframe_4326),
        ]
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert len(result) == 1
        assert mock_from_arrow.call_count == 2

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_stops_after_max_attempts(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_reader.side_effect = OSError("connection reset")
        with pytest.raises(RuntimeError):
            get_overture_features(sample_geodataframe_4326, kind="water")
        assert mock_reader.call_count == MAX_ATTEMPTS

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_backoff_grows_between_attempts(
        self,
        mock_reader,
        mock_from_arrow,
        sample_geodataframe_4326,
        no_backoff_sleep,
    ):
        """A flat retry does not help a throttle that needs time to clear."""
        mock_reader.side_effect = OSError("connection reset")
        with pytest.raises(RuntimeError):
            get_overture_features(sample_geodataframe_4326, kind="water")
        delays = [call.args[0] for call in no_backoff_sleep.call_args_list]
        assert delays == sorted(delays)
        assert len(delays) == MAX_ATTEMPTS - 1

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_backoff_delays_are_2s_then_4s(
        self,
        mock_reader,
        mock_from_arrow,
        sample_geodataframe_4326,
        no_backoff_sleep,
    ):
        """Pinned rather than merely increasing: long enough to clear a
        throttle, short enough that three attempts do not stall a batch run."""
        mock_reader.side_effect = OSError("connection reset")
        with pytest.raises(RuntimeError):
            get_overture_features(sample_geodataframe_4326, kind="water")
        delays = [call.args[0] for call in no_backoff_sleep.call_args_list]
        assert delays == [2.0, 4.0]

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_does_not_sleep_when_the_first_attempt_works(
        self,
        mock_reader,
        mock_from_arrow,
        sample_geodataframe_4326,
        no_backoff_sleep,
    ):
        mock_from_arrow.return_value = _overture_gdf(["lake"], sample_geodataframe_4326)
        get_overture_features(sample_geodataframe_4326, kind="water")
        no_backoff_sleep.assert_not_called()


class TestEmptyBboxIsNotAnError:
    """record_batch_reader returns None both for a genuinely empty bbox and for
    a failed dataset open. Only the second is an error."""

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_returns_empty_when_stac_matches_no_files(
        self, mock_reader, sample_geodataframe_4326, stac_reports_files_present
    ):
        """Open ocean or Antarctica intersects no Overture files at all. That is
        an empty answer, not a failure - raising would cost the scene its water,
        road and building targets alike."""
        mock_reader.return_value = None
        stac_reports_files_present.return_value = None
        result = get_overture_features(sample_geodataframe_4326, kind="buildings")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_does_not_retry_a_genuinely_empty_bbox(
        self, mock_reader, sample_geodataframe_4326, stac_reports_files_present
    ):
        mock_reader.return_value = None
        stac_reports_files_present.return_value = None
        get_overture_features(sample_geodataframe_4326, kind="buildings")
        assert mock_reader.call_count == 1

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_raises_when_files_exist_but_no_reader_is_returned(
        self, mock_reader, sample_geodataframe_4326, stac_reports_files_present
    ):
        mock_reader.return_value = None
        stac_reports_files_present.return_value = ("dataset", None)
        with pytest.raises(RuntimeError, match="failed after"):
            get_overture_features(sample_geodataframe_4326, kind="water")

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_treats_an_unusable_coverage_check_as_an_error(
        self, mock_reader, sample_geodataframe_4326, stac_reports_files_present
    ):
        """_prepare_query is private to overturemaps. If a release breaks it, a
        None reader must stay an error rather than become a silent empty."""
        mock_reader.return_value = None
        stac_reports_files_present.side_effect = AttributeError("gone")
        with pytest.raises(RuntimeError, match="failed after"):
            get_overture_features(sample_geodataframe_4326, kind="water")

    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_treats_a_missing_coverage_check_as_an_error(
        self, mock_reader, sample_geodataframe_4326
    ):
        """The other branch of the same guard: a release that removes or renames
        _prepare_query outright, rather than one where calling it raises."""
        import omniwatermask.overture_source as mod

        mock_reader.return_value = None
        with patch.object(mod.core, "_prepare_query", None):
            with pytest.raises(RuntimeError, match="failed after"):
                get_overture_features(sample_geodataframe_4326, kind="water")


class TestLandClassFiltering:
    """Overture files capes, blowholes and shoals under the water theme.

    They are landforms, so rasterizing them as positive water targets marks
    land as water. OSM_water_tags never selected them either.
    """

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    @pytest.mark.parametrize("land_class", ["cape", "blowhole", "shoal"])
    def test_drops_land_classes(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326, land_class
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["physical", "lake"],
            sample_geodataframe_4326,
            classes=[land_class, "lake"],
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert land_class not in set(result["class"])
        assert set(result["class"]) == {"lake"}

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_keeps_water_classes_of_the_physical_subtype(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """bay/strait/sound are genuine water and must survive the filter -
        dropping the whole "physical" subtype would lose them."""
        mock_from_arrow.return_value = _overture_gdf(
            ["physical", "physical", "physical"],
            sample_geodataframe_4326,
            classes=["bay", "strait", "cape"],
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert set(result["class"]) == {"bay", "strait"}

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_land_classes_kept_for_non_water_kinds(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """The filter is water-specific; a building class collision must not
        silently drop buildings."""
        mock_from_arrow.return_value = _overture_gdf(
            ["residential", "commercial"],
            sample_geodataframe_4326,
            classes=["cape", "shoal"],
        )
        result = get_overture_features(sample_geodataframe_4326, kind="buildings")
        assert len(result) == 2

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_returns_empty_when_only_land_classes(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        mock_from_arrow.return_value = _overture_gdf(
            ["physical", "physical"],
            sample_geodataframe_4326,
            classes=["cape", "shoal"],
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert len(result) == 0

    @patch("omniwatermask.overture_source.gpd.GeoDataFrame.from_arrow")
    @patch("omniwatermask.overture_source.core.record_batch_reader")
    def test_handles_null_classes(
        self, mock_reader, mock_from_arrow, sample_geodataframe_4326
    ):
        """Overture leaves class null on many features; isin(None) must not
        drop them."""
        mock_from_arrow.return_value = _overture_gdf(
            ["lake", "pond"], sample_geodataframe_4326, classes=[None, "pond"]
        )
        result = get_overture_features(sample_geodataframe_4326, kind="water")
        assert len(result) == 2
