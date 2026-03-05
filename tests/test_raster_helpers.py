import numpy as np
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box

from omniwatermask.raster_helpers import (
    export_to_disk,
    rasterize_vector,
    resample_input,
)


class TestResampleInput:
    def test_resamples_to_lower_resolution(self, sample_geotiff, tmp_dir):
        """Resampling to a coarser resolution should produce a smaller raster."""
        result = resample_input(sample_geotiff, resample_res=20, output_dir=tmp_dir)
        assert result.exists()
        with rio.open(result) as src:
            # Original is 10m/px (1000m / 100px), resample to 20m → 50px
            assert src.width == 50
            assert src.height == 50

    def test_resamples_to_higher_resolution(self, sample_geotiff, tmp_dir):
        """Resampling to a finer resolution should produce a larger raster."""
        result = resample_input(sample_geotiff, resample_res=5, output_dir=tmp_dir)
        with rio.open(result) as src:
            assert src.width == 200
            assert src.height == 200

    def test_returns_cached_if_exists(self, sample_geotiff, tmp_dir):
        """If the resampled file already exists, it should return early."""
        result1 = resample_input(sample_geotiff, resample_res=20, output_dir=tmp_dir)
        mtime1 = result1.stat().st_mtime
        result2 = resample_input(sample_geotiff, resample_res=20, output_dir=tmp_dir)
        assert result1 == result2
        assert result2.stat().st_mtime == mtime1  # file was not rewritten

    def test_preserves_crs(self, sample_geotiff, tmp_dir):
        result = resample_input(sample_geotiff, resample_res=20, output_dir=tmp_dir)
        with rio.open(result) as src, rio.open(sample_geotiff) as orig:
            assert src.crs == orig.crs

    def test_preserves_band_count(self, sample_geotiff, tmp_dir):
        result = resample_input(sample_geotiff, resample_res=20, output_dir=tmp_dir)
        with rio.open(result) as src, rio.open(sample_geotiff) as orig:
            assert src.count == orig.count


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
