"""End-to-end tests for OmniWaterMask.

These tests exercise the full pipeline with real raster data (NAIP)
and real model inference. No mocking — all components run for real.

A small crop of a NAIP water scene is downloaded on first run and
cached in the tests/data/ directory. Model weights are downloaded
from HuggingFace and cached by platformdirs.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.windows import Window
from shapely.geometry import box

from omniwatermask import make_water_mask, make_water_mask_debug
from omniwatermask.raster_helpers import (
    export_to_disk,
    rasterize_vector,
    resample_input,
)
from omniwatermask.vector_cache import (
    add_to_db,
    check_db,
    initialize_db,
)
from omniwatermask.water_inf_helpers import (
    get_NDWI,
    integrate_water_detection_methods,
)
from omniwatermask.water_inf_pipeline import collect_models


pytestmark = pytest.mark.e2e

NAIP_URL = (
    "https://huggingface.co/datasets/giswqs/geospatial"
    "/resolve/main/naip/naip_water_test.tif"
)
# NAIP band order is R, G, B, NIR (bands 1-4)
# OmniWaterMask expects [Red, Green, Blue, NIR]
NAIP_BAND_ORDER = [1, 2, 3, 4]
# Crop window: 256x256 from the centre of the image
# (the image is 5390x7580, centre ~2695,3790)
CROP_WINDOW = Window(col_off=2100, row_off=3500, width=256, height=256)
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def naip_crop():
    """Download a small crop of real NAIP imagery, cached on disk.

    The full NAIP scene is ~157MB. We use GDAL vsicurl to read
    just a 256x256 window remotely and save the crop locally.
    Subsequent test runs reuse the cached crop.
    """
    DATA_DIR.mkdir(exist_ok=True)
    crop_path = DATA_DIR / "naip_crop_256.tif"

    if not crop_path.exists():
        vsicurl_path = f"/vsicurl/{NAIP_URL}"
        with rio.open(vsicurl_path) as src:
            data = src.read(window=CROP_WINDOW)
            transform = src.window_transform(CROP_WINDOW)
            profile = src.profile.copy()
            profile.update(
                height=CROP_WINDOW.height,
                width=CROP_WINDOW.width,
                transform=transform,
            )
        with rio.open(crop_path, "w", **profile) as dst:
            dst.write(data)

    return crop_path


@pytest.fixture(scope="session")
def loaded_models():
    """Download and load real model weights (cached across tests)."""
    device = torch.device("cpu")
    dtype = torch.float32
    models = collect_models(
        model_path="",
        destination_model_dir=None,
        model_download_source="hugging_face",
        inference_device=device,
        inference_dtype=dtype,
    )
    return models


class TestResampleAndExportRoundTrip:
    """Resample -> export -> read preserves spatial properties."""

    def test_resample_then_export(self, naip_crop, tmp_path):
        resampled = resample_input(naip_crop, resample_res=2, output_dir=tmp_path)

        with rio.open(resampled) as src:
            assert src.width == 128
            assert src.height == 128

        output = np.ones((1, 128, 128), dtype=np.uint8)
        export_path = tmp_path / "mask.tif"
        export_to_disk(output, export_path, resampled, ["water_mask"])

        with rio.open(export_path) as dst:
            assert dst.crs is not None
            assert dst.width == 128
            assert dst.height == 128
            assert dst.descriptions == ("water_mask",)


class TestVectorCacheRoundTrip:
    """Full cache workflow: init -> add -> check -> view."""

    def test_full_cache_cycle(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        initialize_db(cache_dir)

        polygon = box(0, 0, 1, 1)
        paths = [Path("/some/raster.tif")]
        gdf = gpd.GeoDataFrame(geometry=[box(0.1, 0.1, 0.9, 0.9)], crs="EPSG:4326")

        _, found = check_db(cache_dir, polygon, paths, water=True)
        assert found is False

        add_to_db(cache_dir, polygon, paths, gdf, water=True)

        result_gdf, found = check_db(cache_dir, polygon, paths, water=True)
        assert found is True
        assert len(result_gdf) == 1

        _, found = check_db(cache_dir, polygon, paths, water=False, roads=True)
        assert found is False


class TestRasterizeVectorIntegration:
    """Rasterize vector data against real raster profiles."""

    def test_rasterize_water_body_polygon(self, naip_crop):
        with rio.open(naip_crop) as src:
            bounds = src.bounds
            profile = src.profile

        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        water_poly = box(cx - 50, cy - 50, cx + 50, cy + 50)
        gdf = gpd.GeoDataFrame(geometry=[water_poly], crs=profile["crs"])

        mask = rasterize_vector(gdf, profile)
        assert mask.shape == (256, 256)
        assert mask[128, 128] == 1
        assert mask[0, 0] == 0
        coverage = mask.sum() / mask.size
        assert 0.01 < coverage < 0.5


class TestNDWIOnRealData:
    """NDWI calculation on real NAIP imagery."""

    def test_ndwi_produces_valid_range(self, naip_crop):
        with rio.open(naip_crop) as src:
            bands = src.read(NAIP_BAND_ORDER).astype(np.float32)

        ndwi = get_NDWI(bands, "cpu")
        assert ndwi.shape == (256, 256)
        # NDWI should be in [-1, 1] range (ignoring nodata/div-by-zero)
        valid = ~torch.isnan(ndwi) & ~torch.isinf(ndwi)
        assert (ndwi[valid] >= -1).all()
        assert (ndwi[valid] <= 1).all()


class TestIntegrateWaterDetectionReal:
    """Full integration pipeline with real model inference."""

    def test_real_inference_on_naip(self, naip_crop, loaded_models, tmp_path):
        """Real model + NDWI on real NAIP raster, no OSM."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with rio.open(naip_crop) as src:
            bands = src.read(NAIP_BAND_ORDER).astype(np.float32)

        result, layer_names = integrate_water_detection_methods(
            input_bands=bands,
            input_path=naip_crop,
            cache_dir=cache_dir,
            inference_dtype=torch.float32,
            inference_device=torch.device("cpu"),
            inference_patch_size=256,
            inference_overlap_size=0,
            batch_size=1,
            models=loaded_models,
            use_osm_water=False,
            use_ndwi=True,
            use_model=True,
            use_osm_building_mask=False,
            use_osm_roads_mask=False,
            mosaic_device="cpu",
            use_cache=False,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 256, 256)
        assert "Water predictions" in layer_names
        assert "No data mask" in layer_names
        # No-data mask should be all valid (no zeros in NAIP)
        assert result[1].sum() > 0

    def test_real_debug_output(self, naip_crop, loaded_models, tmp_path):
        """Debug output should contain all diagnostic layers."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with rio.open(naip_crop) as src:
            bands = src.read(NAIP_BAND_ORDER).astype(np.float32)

        result, layer_names = integrate_water_detection_methods(
            input_bands=bands,
            input_path=naip_crop,
            cache_dir=cache_dir,
            inference_dtype=torch.float32,
            inference_device=torch.device("cpu"),
            inference_patch_size=256,
            inference_overlap_size=0,
            batch_size=1,
            models=loaded_models,
            use_osm_water=False,
            use_ndwi=True,
            use_model=True,
            use_osm_building_mask=False,
            use_osm_roads_mask=False,
            mosaic_device="cpu",
            use_cache=False,
            debug_output=True,
        )

        assert result.ndim == 3
        assert len(layer_names) > 2
        assert "Water predictions" in layer_names
        assert "NDWI binary" in layer_names
        assert "Model confidence" in layer_names


class TestFullPipelineReal:
    """Full make_water_mask pipeline with real inference on NAIP."""

    def test_produces_output_file(self, naip_crop, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        output_paths = make_water_mask(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
        )

        assert len(output_paths) == 1
        assert output_paths[0].exists()
        with rio.open(output_paths[0]) as src:
            assert src.count == 2
            assert src.height == 256
            assert src.width == 256
            water = src.read(1)
            assert water.dtype == np.uint8
            # Water predictions should be binary
            assert set(np.unique(water)).issubset({0, 1})

    def test_skips_existing_when_overwrite_false(self, naip_crop, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        output_paths = make_water_mask(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        first_mtime = output_paths[0].stat().st_mtime

        output_paths2 = make_water_mask(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            overwrite=False,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
        )

        assert output_paths2[0].stat().st_mtime == first_mtime

    def test_debug_mode(self, naip_crop, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        output_paths = make_water_mask_debug(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            debug_output=True,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
        )

        assert len(output_paths) == 1
        assert output_paths[0].exists()
        with rio.open(output_paths[0]) as src:
            assert src.count > 2


class TestPipelineConfigurations:
    """Run the pipeline with various input configurations."""

    def _run_pipeline(self, naip_crop, tmp_path, **kwargs):
        """Helper to run make_water_mask_debug with defaults."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)

        defaults = dict(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
            overwrite=True,
        )
        defaults.update(kwargs)
        return make_water_mask_debug(**defaults)

    def _validate_output(self, output_paths, expected_count=2):
        assert len(output_paths) >= 1
        for p in output_paths:
            assert p.exists()
            with rio.open(p) as src:
                assert src.count >= expected_count
                assert src.height > 0
                assert src.width > 0

    def test_model_only_no_ndwi(self, naip_crop, tmp_path):
        """Model inference without NDWI (requires vector targets)."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            use_ndwi=False,
            use_osm_water=True,
        )
        self._validate_output(paths)

    def test_ndwi_only_no_model(self, naip_crop, tmp_path):
        """NDWI-only detection without model (requires vector targets)."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            use_model=False,
            use_osm_water=True,
        )
        self._validate_output(paths)

    def test_model_without_optimisation(self, naip_crop, tmp_path):
        """Model + NDWI but with optimise_model=False."""
        paths = self._run_pipeline(naip_crop, tmp_path, optimise_model=False)
        self._validate_output(paths)

    def test_with_resampling(self, naip_crop, tmp_path):
        """Pipeline with resampling to 2m resolution."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            resample_res=2,
            inference_patch_size=128,
        )
        self._validate_output(paths)
        with rio.open(paths[0]) as src:
            assert src.height == 128
            assert src.width == 128

    def test_with_cache_enabled(self, naip_crop, tmp_path):
        """Pipeline with vector caching enabled."""
        paths = self._run_pipeline(naip_crop, tmp_path, use_cache=True)
        self._validate_output(paths)

    def test_with_cache_disabled(self, naip_crop, tmp_path):
        """Pipeline with vector caching disabled."""
        paths = self._run_pipeline(naip_crop, tmp_path, use_cache=False)
        self._validate_output(paths)

    def test_string_scene_path(self, naip_crop, tmp_path):
        """Accept a string path instead of Path object."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            scene_paths=[str(naip_crop)],
        )
        self._validate_output(paths)

    def test_single_path_not_list(self, naip_crop, tmp_path):
        """Accept a single Path (not wrapped in a list)."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            scene_paths=naip_crop,
        )
        self._validate_output(paths)

    def test_single_string_path(self, naip_crop, tmp_path):
        """Accept a single string path."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            scene_paths=str(naip_crop),
        )
        self._validate_output(paths)

    def test_debug_output_all_methods(self, naip_crop, tmp_path):
        """Debug output with both model and NDWI enabled."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            debug_output=True,
            use_model=True,
            use_ndwi=True,
        )
        self._validate_output(paths, expected_count=5)

    def test_debug_output_model_only(self, naip_crop, tmp_path):
        """Debug output with model only (requires vector targets)."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            debug_output=True,
            use_model=True,
            use_ndwi=False,
            use_osm_water=True,
        )
        self._validate_output(paths, expected_count=3)

    def test_debug_output_ndwi_only(self, naip_crop, tmp_path):
        """Debug output with NDWI only (requires vector targets)."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            debug_output=True,
            use_model=False,
            use_ndwi=True,
            use_osm_water=True,
        )
        self._validate_output(paths, expected_count=3)

    def test_smaller_patch_with_overlap(self, naip_crop, tmp_path):
        """Smaller inference patches with overlap."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            inference_patch_size=128,
            inference_overlap_size=32,
        )
        self._validate_output(paths)

    def test_multiple_scenes(self, naip_crop, tmp_path):
        """Process the same scene twice as multiple inputs."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            scene_paths=[naip_crop, naip_crop],
        )
        assert len(paths) == 2
        self._validate_output(paths)

    def test_float16_inference(self, naip_crop, tmp_path):
        """Run inference with float16 dtype."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            inference_dtype=torch.float16,
        )
        self._validate_output(paths)

    def test_custom_no_data_value(self, naip_crop, tmp_path):
        """Pipeline with a custom no_data_value."""
        paths = self._run_pipeline(
            naip_crop,
            tmp_path,
            no_data_value=255,
        )
        self._validate_output(paths)

    def test_make_water_mask_default_api(self, naip_crop, tmp_path):
        """Use the public make_water_mask API (non-debug)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        paths = make_water_mask(
            scene_paths=[naip_crop],
            band_order=NAIP_BAND_ORDER,
            output_dir=output_dir,
            use_osm_water=False,
            use_osm_building=False,
            use_osm_roads=False,
            cache_dir=cache_dir,
            inference_patch_size=256,
            inference_overlap_size=0,
        )
        self._validate_output(paths)
        with rio.open(paths[0]) as src:
            water = src.read(1)
            assert set(np.unique(water)).issubset({0, 1})
