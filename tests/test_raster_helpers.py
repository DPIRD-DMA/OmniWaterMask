from pathlib import Path

import pytest
import numpy as np
import torch
import geopandas as gpd
import rasterio as rio
import rasterio.shutil
from shapely.geometry import box

from omniwatermask.raster_helpers import (
    export_to_disk,
    rasterize_vector,
    resample_input,
)


class TestResampleInput:
    def test_resamples_to_lower_resolution(self, sample_geotiff):
        """Resampling to a coarser resolution should produce a smaller raster."""
        result = resample_input(sample_geotiff, resample_res=20)
        with rio.open(result) as src:
            # Original is 10m/px (1000m / 100px), resample to 20m → 50px
            assert src.width == 50
            assert src.height == 50
        rasterio.shutil.delete(result)

    def test_resamples_to_higher_resolution(self, sample_geotiff):
        """Resampling to a finer resolution should produce a larger raster."""
        result = resample_input(sample_geotiff, resample_res=5)
        with rio.open(result) as src:
            assert src.width == 200
            assert src.height == 200
        rasterio.shutil.delete(result)

    def test_writes_nothing_to_disk(self, sample_geotiff):
        """The resampled scene is a /vsimem VRT, not a raster next to the input."""
        before = set(sample_geotiff.parent.iterdir())
        result = resample_input(sample_geotiff, resample_res=20)
        assert result.startswith("/vsimem/")
        assert not Path(result).exists()
        assert set(sample_geotiff.parent.iterdir()) == before
        rasterio.shutil.delete(result)

    def test_each_call_gets_its_own_path(self, sample_geotiff):
        """Two live resamples must not collide in the shared /vsimem namespace."""
        first = resample_input(sample_geotiff, resample_res=20)
        second = resample_input(sample_geotiff, resample_res=20)
        assert first != second
        rasterio.shutil.delete(first)
        # Deleting one must leave the other readable.
        with rio.open(second) as src:
            assert src.width == 50
        rasterio.shutil.delete(second)

    def test_readable_after_source_dataset_closed(self, sample_geotiff):
        """The VRT resolves its source itself, long after resample_input returns."""
        result = resample_input(sample_geotiff, resample_res=20)
        with rio.open(result) as src, rio.open(sample_geotiff) as orig:
            expected = orig.read([1, 2], out_shape=(2, 50, 50))
            assert np.array_equal(src.read([1, 2]), expected)
        rasterio.shutil.delete(result)

    def test_preserves_crs(self, sample_geotiff):
        result = resample_input(sample_geotiff, resample_res=20)
        with rio.open(result) as src, rio.open(sample_geotiff) as orig:
            assert src.crs == orig.crs
        rasterio.shutil.delete(result)

    def test_preserves_band_count(self, sample_geotiff):
        result = resample_input(sample_geotiff, resample_res=20)
        with rio.open(result) as src, rio.open(sample_geotiff) as orig:
            assert src.count == orig.count
        rasterio.shutil.delete(result)

    def test_resolves_relative_source_paths(self, sample_geotiff, monkeypatch):
        """A VRT outlives the cwd it was built in, so the source must be absolute."""
        monkeypatch.chdir(sample_geotiff.parent)
        result = resample_input(Path(sample_geotiff.name), resample_res=20)
        monkeypatch.chdir(Path(__file__).parent)
        with rio.open(result) as src:
            assert src.read(1).shape == (50, 50)
        rasterio.shutil.delete(result)


class TestExportToDisk:
    def test_exports_valid_geotiff(self, sample_geotiff, tmp_dir):
        array = np.random.rand(2, 100, 100).astype(np.float32)
        export_path = tmp_dir / "output.tif"
        export_to_disk(
            array=array,
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["layer1", "layer2"],
        )
        assert export_path.exists()
        with rio.open(export_path) as src:
            assert src.count == 2
            assert src.height == 100
            assert src.width == 100
            assert src.descriptions == ("layer1", "layer2")

    def test_nodata_mask_written_as_gdal_mask(self, sample_geotiff, tmp_dir):
        array = np.ones((1, 100, 100), dtype=np.uint8)
        nodata_mask = np.ones((100, 100), dtype=np.uint8)
        nodata_mask[:10, :10] = 0  # a no-data corner
        export_path = tmp_dir / "masked.tif"
        export_to_disk(
            array=array,
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["Water predictions"],
            nodata_mask=nodata_mask,
        )
        # mask is embedded in the GeoTIFF, not a sidecar file
        assert not (tmp_dir / "masked.tif.msk").exists()
        with rio.open(export_path) as src:
            # mask is a dataset mask, not an extra data band
            assert src.count == 1
            mask = src.read_masks(1)
            assert mask[0, 0] == 0  # no-data corner
            assert mask[50, 50] == 255  # valid

    def test_preserves_crs_and_transform(self, sample_geotiff, tmp_dir):
        array = np.ones((1, 100, 100), dtype=np.float32)
        export_path = tmp_dir / "output.tif"
        export_to_disk(array, export_path, sample_geotiff, ["test"])
        with rio.open(export_path) as dst, rio.open(sample_geotiff) as src:
            assert dst.crs == src.crs
            assert dst.transform == src.transform

    def test_single_band_export(self, sample_geotiff, tmp_dir):
        array = np.zeros((1, 100, 100), dtype=np.uint8)
        export_path = tmp_dir / "single.tif"
        export_to_disk(array, export_path, sample_geotiff, ["mask"])
        with rio.open(export_path) as src:
            assert src.count == 1
            data = src.read(1)
            assert np.all(data == 0)


class TestRasterizeVector:
    def test_rasterizes_polygon(self, sample_rasterio_src, sample_geodataframe):
        profile = sample_rasterio_src.profile
        gdf = sample_geodataframe.to_crs(sample_rasterio_src.crs)
        result = rasterize_vector(gdf, profile)
        assert result.shape == (100, 100)
        assert result.dtype == rio.uint8
        # The polygon covers the interior, so there should be 1s
        assert result.sum() > 0
        # Edges should remain 0 (polygon doesn't cover full extent)
        assert result[0, 0] == 0

    def test_empty_geodataframe_returns_zeros(self, sample_rasterio_src):
        profile = sample_rasterio_src.profile
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32650")
        result = rasterize_vector(empty_gdf, profile)
        assert result.shape == (100, 100)
        assert result.sum() == 0

    def test_full_coverage_polygon(self, sample_rasterio_src):
        """A polygon covering the full extent should rasterize to all 1s."""
        profile = sample_rasterio_src.profile
        bounds = sample_rasterio_src.bounds
        full_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf = gpd.GeoDataFrame(geometry=[full_poly], crs=sample_rasterio_src.crs)
        result = rasterize_vector(gdf, profile)
        assert result.sum() == 100 * 100


class TestExportToDiskStreaming:
    """The debug path hands over a list of layers instead of a stacked array."""

    def test_writes_each_layer_as_its_own_band(self, sample_geotiff, tmp_dir):
        layers = [
            torch.full((100, 100), 3, dtype=torch.uint8),
            torch.ones((100, 100), dtype=torch.bool),
        ]
        export_path = tmp_dir / "streamed.tif"
        export_to_disk(
            array=layers,
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["counts", "flags"],
        )
        with rio.open(export_path) as src:
            assert src.count == 2
            assert src.dtypes == ("float32", "float32")
            assert src.descriptions == ("counts", "flags")
            assert np.all(src.read(1) == 3.0)
            assert np.all(src.read(2) == 1.0)

    def test_releases_each_layer_as_it_is_written(self, sample_geotiff, tmp_dir):
        """The memory contract: a written layer must not still be referenced.

        Holding all 14 debug layers as float32 plus a stacked copy is what cost
        12.6 GB on a full Sentinel-2 tile. Streaming only helps if each source
        is actually dropped, which callers rely on.
        """
        layers = [torch.ones((50, 50)), torch.zeros((50, 50))]
        export_to_disk(
            array=layers,
            export_path=tmp_dir / "released.tif",
            source_path=sample_geotiff,
            layer_names=["a", "b"],
        )
        assert layers == [None, None]

    def test_none_layer_is_written_as_zeros(self, sample_geotiff, tmp_dir):
        layers = [torch.ones((100, 100)), None]
        export_path = tmp_dir / "with_none.tif"
        export_to_disk(
            array=layers,
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["present", "missing"],
        )
        with rio.open(export_path) as src:
            assert np.all(src.read(2) == 0.0)

    def test_uses_band_interleave_and_tiles(self, sample_geotiff, tmp_dir):
        """Band-at-a-time writes only compress well with BAND interleave."""
        export_path = tmp_dir / "layout.tif"
        export_to_disk(
            array=[torch.ones((100, 100)), torch.zeros((100, 100))],
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["a", "b"],
        )
        with rio.open(export_path) as src:
            assert src.profile["tiled"] is True
            assert src.interleaving.name.lower() == "band"

    def test_all_none_is_rejected(self, sample_geotiff, tmp_dir):
        with pytest.raises(ValueError):
            export_to_disk(
                array=[None, None],
                export_path=tmp_dir / "empty.tif",
                source_path=sample_geotiff,
                layer_names=["a", "b"],
            )

    def test_stacked_array_path_is_unchanged(self, sample_geotiff, tmp_dir):
        """The single-band mask still goes through as a numpy array."""
        array = np.ones((1, 100, 100), dtype=np.uint8)
        export_path = tmp_dir / "mask.tif"
        export_to_disk(
            array=array,
            export_path=export_path,
            source_path=sample_geotiff,
            layer_names=["Water predictions"],
        )
        with rio.open(export_path) as src:
            assert src.count == 1
            assert src.dtypes == ("uint8",)
