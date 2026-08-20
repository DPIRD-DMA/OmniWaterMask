"""Fast orchestration tests with the model mocked out.

The two functions that tie OmniWaterMask together —
``integrate_water_detection_methods`` and ``make_water_mask`` — run real
model inference end to end, which is too slow for CI (see the ``e2e``
tests for the real thing). Here we monkeypatch ``predict_from_array`` to
return a synthetic confidence array so the surrounding orchestration
(NDWI, vector-target threading, multi-scale optimisation, stacking and
export) runs for real on the tiny synthetic raster from ``conftest`` in
well under a second. These tests are intentionally *not* marked ``e2e``
so they run in CI and guard against wiring regressions.
"""

import numpy as np
import pytest
import rasterio as rio
import torch

from omniwatermask import make_water_mask, make_water_mask_debug
from omniwatermask.target_builders import TargetBuildError
from omniwatermask.water_inf_helpers import integrate_water_detection_methods

# Synthetic raster band order (conftest writes Blue, Green, Red, NIR)
BAND_ORDER = [1, 2, 3, 4]


def _fake_confidence(input_array, **kwargs):
    """Stand in for ``predict_from_array``.

    Returns a 2-class softmax-style confidence array shaped like the
    input ((2, H, W)): class 0 = land, class 1 = water, with water
    concentrated in the centre so the downstream thresholding has a real
    cluster to optimise.
    """
    _, height, width = input_array.shape
    water = np.zeros((height, width), dtype=np.float32)
    water[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 0.9
    land = 1.0 - water
    return np.stack([land, water])


@pytest.fixture
def mock_model(monkeypatch):
    """Replace real inference (and model loading) with the fake above."""
    monkeypatch.setattr(
        "omniwatermask.water_inf_helpers.predict_from_array", _fake_confidence
    )
    # make_water_mask_debug loads weights via collect_models — skip the
    # download/load entirely since predict_from_array no longer uses them.
    monkeypatch.setattr(
        "omniwatermask.water_inf_pipeline.collect_models",
        lambda **kwargs: ["dummy-model"],
    )


@pytest.fixture
def synthetic_bands(sample_geotiff):
    """Read the synthetic raster as a float32 (4, H, W) array."""
    with rio.open(sample_geotiff) as src:
        return src.read().astype(np.float32)


def _integrate(sample_geotiff, bands, tmp_path, **overrides):
    """Call integrate_water_detection_methods with CPU/no-network defaults."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    kwargs = dict(
        input_bands=bands,
        input_path=sample_geotiff,
        cache_dir=cache_dir,
        inference_dtype=torch.float32,
        inference_device=torch.device("cpu"),
        inference_patch_size=256,
        inference_overlap_size=0,
        batch_size=1,
        models=["dummy-model"],
        use_osm_water=False,
        use_osm_building_mask=False,
        use_osm_roads_mask=False,
        mosaic_device="cpu",
        use_cache=False,
    )
    kwargs.update(overrides)
    return integrate_water_detection_methods(**kwargs)


class TestIntegrateWaterDetectionMethods:
    """Direct tests of the integration layer (model mocked)."""

    def test_model_and_ndwi(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path
    ):
        result, layer_names, nodata_mask = _integrate(
            sample_geotiff, synthetic_bands, tmp_path, use_ndwi=True, use_model=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 100, 100)
        assert layer_names == ["Water predictions"]
        # Water mask is binary uint8
        assert set(np.unique(result[0])).issubset({0, 1})
        # No-data mask: synthetic raster has no all-zero pixels, so all valid
        assert nodata_mask is not None
        assert nodata_mask.sum() > 0

    def test_model_only(self, mock_model, sample_geotiff, synthetic_bands, tmp_path):
        result, layer_names, _ = _integrate(
            sample_geotiff, synthetic_bands, tmp_path, use_ndwi=False, use_model=True
        )
        assert result.shape == (1, 100, 100)
        assert "Water predictions" in layer_names

    def test_ndwi_only(self, mock_model, sample_geotiff, synthetic_bands, tmp_path):
        result, layer_names, _ = _integrate(
            sample_geotiff, synthetic_bands, tmp_path, use_ndwi=True, use_model=False
        )
        assert result.shape == (1, 100, 100)
        assert "Water predictions" in layer_names

    def test_debug_output_has_diagnostic_layers(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path
    ):
        result, layer_names, _ = _integrate(
            sample_geotiff,
            synthetic_bands,
            tmp_path,
            use_ndwi=True,
            use_model=True,
            debug_output=True,
        )
        # debug output is handed on as a list of layers, promoted per band at
        # export time rather than stacked here
        assert isinstance(result, list)
        assert len(result) == len(layer_names)
        assert len(layer_names) > 2
        assert "Water predictions" in layer_names
        assert "NDWI binary" in layer_names
        assert "Model confidence" in layer_names

    def test_without_model_optimisation(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path
    ):
        result, _, _ = _integrate(
            sample_geotiff,
            synthetic_bands,
            tmp_path,
            use_ndwi=True,
            use_model=True,
            optimise_model=False,
        )
        assert result.shape == (1, 100, 100)


class TestMakeWaterMaskPipeline:
    """Tests of the full make_water_mask / _debug pipeline (model mocked)."""

    def test_produces_binary_output_file(self, mock_model, sample_geotiff, tmp_path):
        output_dir = tmp_path / "out"
        output_paths = make_water_mask(
            scene_paths=[sample_geotiff],
            band_order=BAND_ORDER,
            output_dir=output_dir,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=tmp_path / "cache",
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        assert len(output_paths) == 1
        assert output_paths[0].exists()
        with rio.open(output_paths[0]) as src:
            assert src.count == 1
            assert src.height == 100
            assert src.width == 100
            water = src.read(1)
            assert water.dtype == np.uint8
            assert set(np.unique(water)).issubset({0, 1})
            # No-data written as a GDAL mask, not a data band
            mask = src.read_masks(1)
            assert mask.shape == (100, 100)

    def test_debug_output_has_extra_bands(self, mock_model, sample_geotiff, tmp_path):
        output_dir = tmp_path / "out"
        output_paths = make_water_mask_debug(
            scene_paths=sample_geotiff,
            band_order=BAND_ORDER,
            output_dir=output_dir,
            debug_output=True,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=tmp_path / "cache",
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        assert len(output_paths) == 1
        with rio.open(output_paths[0]) as src:
            assert src.count > 2

    def test_skips_existing_when_overwrite_false(
        self, mock_model, sample_geotiff, tmp_path
    ):
        kwargs = dict(
            scene_paths=[sample_geotiff],
            band_order=BAND_ORDER,
            output_dir=tmp_path / "out",
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=tmp_path / "cache",
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        first = make_water_mask(**kwargs)
        first_mtime = first[0].stat().st_mtime

        second = make_water_mask(overwrite=False, **kwargs)
        assert second[0].stat().st_mtime == first_mtime

    def test_accepts_string_path(self, mock_model, sample_geotiff, tmp_path):
        output_paths = make_water_mask(
            scene_paths=str(sample_geotiff),
            band_order=BAND_ORDER,
            output_dir=tmp_path / "out",
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=tmp_path / "cache",
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        assert output_paths[0].exists()

    def test_requires_target_or_model_and_ndwi(
        self, mock_model, sample_geotiff, tmp_path
    ):
        """Without vector targets, both model and NDWI must be enabled."""
        with pytest.raises(ValueError):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=BAND_ORDER,
                output_dir=tmp_path / "out",
                use_osm_water=False,
                use_model=False,
                cache_dir=tmp_path / "cache",
            )


class TestTargetThreadDatasetHandles:
    """The target threads read from datasets opened by the integration layer.

    ``build_targets`` does not close what it is handed - it is also called
    directly with a dataset its caller goes on to reuse - so the integration
    layer owns those handles. Leaking one per scene turns a large batch into
    file-descriptor exhaustion, so they must be closed however the scene ends.
    """

    @staticmethod
    def _record_opens(monkeypatch):
        """Collect every dataset the integration layer opens."""
        opened = []
        real_open = rio.open

        def recording_open(*args, **kwargs):
            src = real_open(*args, **kwargs)
            opened.append(src)
            return src

        monkeypatch.setattr("omniwatermask.water_inf_helpers.rio.open", recording_open)
        return opened

    @staticmethod
    def _stub_build_targets(monkeypatch, result=None, error=None):
        """Replace the real fetch so no network is touched."""

        def fake_build_targets(queue=None, **kwargs):
            outcome = error if error is not None else result
            if queue is not None:
                queue.put(outcome)
                return queue
            if error is not None:
                raise error
            return result

        monkeypatch.setattr(
            "omniwatermask.water_inf_helpers.build_targets", fake_build_targets
        )

    def _integrate_with_targets(self, sample_geotiff, bands, tmp_path):
        """Both target threads on, so both handles are opened."""
        return _integrate(
            sample_geotiff,
            bands,
            tmp_path,
            use_osm_water=True,
            use_osm_building_mask=True,
            use_osm_roads_mask=True,
        )

    def test_closed_after_a_successful_run(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path, monkeypatch
    ):
        opened = self._record_opens(monkeypatch)
        self._stub_build_targets(monkeypatch)
        self._integrate_with_targets(sample_geotiff, synthetic_bands, tmp_path)

        assert len(opened) == 2, "expected one handle per target thread"
        assert all(src.closed for src in opened)

    def test_closed_when_a_target_build_fails(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path, monkeypatch
    ):
        """The scene is skipped on TargetBuildError, but not at the cost of a
        leaked handle - a widespread outage would leak one per scene."""
        opened = self._record_opens(monkeypatch)
        self._stub_build_targets(monkeypatch, error=TargetBuildError("no vectors"))

        with pytest.raises(TargetBuildError):
            self._integrate_with_targets(sample_geotiff, synthetic_bands, tmp_path)

        assert len(opened) == 2
        assert all(src.closed for src in opened)

    def test_closed_when_model_inference_raises(
        self, mock_model, sample_geotiff, synthetic_bands, tmp_path, monkeypatch
    ):
        """Inference runs while the target threads are still going, so a failure
        there is the case the handles are most likely to escape from."""
        opened = self._record_opens(monkeypatch)
        self._stub_build_targets(monkeypatch)

        def exploding_inference(*args, **kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(
            "omniwatermask.water_inf_helpers.predict_from_array", exploding_inference
        )

        with pytest.raises(RuntimeError, match="out of memory"):
            self._integrate_with_targets(sample_geotiff, synthetic_bands, tmp_path)

        assert len(opened) == 2
        assert all(src.closed for src in opened)
