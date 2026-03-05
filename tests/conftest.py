
import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_bounds
from shapely.geometry import box


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a minimal 4-band GeoTIFF in EPSG:32650 (UTM zone 50N).

    The raster covers a small area near (115.86, -31.95) — Perth, Australia.
    Bands simulate Blue, Green, Red, NIR with water-like spectral signature
    in the centre and land around the edges.
    """
    path = tmp_path / "sample.tif"
    height, width = 100, 100
    # Bounds in EPSG:32650 (roughly 115.86E, -31.95S)
    left, bottom, right, top = 390000.0, 6460000.0, 391000.0, 6461000.0
    transform = from_bounds(left, bottom, right, top, width, height)

    bands = np.zeros((4, height, width), dtype=np.uint16)
    # Land signature: higher NIR than green
    bands[0, :, :] = 500   # Blue
    bands[1, :, :] = 600   # Green
    bands[2, :, :] = 700   # Red
    bands[3, :, :] = 1200  # NIR

    # Water signature in centre 40x40: higher green than NIR
    bands[0, 30:70, 30:70] = 300   # Blue
    bands[1, 30:70, 30:70] = 400   # Green
    bands[2, 30:70, 30:70] = 200   # Red
    bands[3, 30:70, 30:70] = 100   # NIR

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 4,
        "height": height,
        "width": width,
        "crs": "EPSG:32650",
        "transform": transform,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(bands)
        dst.descriptions = ("Blue", "Green", "Red", "NIR")

    return path


@pytest.fixture
def sample_rasterio_src(sample_geotiff):
    """Open the sample GeoTIFF as a rasterio DatasetReader."""
    src = rio.open(sample_geotiff)
    yield src
    src.close()


@pytest.fixture
def sample_geodataframe():
    """Create a simple GeoDataFrame with a polygon in EPSG:32650."""
    poly = box(390200, 6460200, 390800, 6460800)
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32650")


@pytest.fixture
def sample_geodataframe_4326(sample_rasterio_src):
    """Create a GeoDataFrame of the raster bounds in EPSG:4326."""
    src = sample_rasterio_src
    bounds = src.bounds
    bbox_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=src.crs)
    return gdf.to_crs("EPSG:4326")


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory for vector_cache tests."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
