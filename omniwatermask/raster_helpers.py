from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features


def resample_input(
    input_path: Path, resample_res: Union[int, float], output_dir: Path
) -> Path:
    with rio.open(input_path) as src:
        resample_path = output_dir / f"{input_path.stem}_resample_{resample_res}m.tif"
        if resample_path.exists():
            return resample_path

        scale_factor = src.res[0] / resample_res
        new_height = round(src.height * scale_factor)
        new_width = round(src.width * scale_factor)

        profile = src.profile.copy()
        profile.update(
            height=new_height,
            width=new_width,
            transform=rio.transform.from_bounds(*src.bounds, new_width, new_height),  # type: ignore
            alpha="unspecified",
        )
        data = src.read(out_shape=(src.count, new_height, new_width))

        with rio.open(resample_path, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = src.descriptions
            dst.colorinterp = src.colorinterp

    return resample_path


def export_to_disk(
    array: Union[np.ndarray, list[np.ndarray]],
    export_path: Path,
    source_path: Path,
    layer_names: list[str],
    mask: Optional[np.ndarray] = None,
):
    """Export the array to disk as a GeoTIFF.

    If `mask` is provided (2-D, 1=valid, 0=nodata), it is written as a GDAL
    internal mask band so QGIS/GDAL render nodata pixels as transparent.
    """
    src = rio.open(source_path)

    if isinstance(array, list):
        count = len(array)
        dtype = array[0].dtype
        height, width = array[0].shape
    else:
        count = array.shape[0]
        dtype = array.dtype
        height, width = array.shape[1], array.shape[2]

    profile = {
        "dtype": dtype,
        "count": count,
        "compress": "lzw",
        "nodata": None,
        "driver": "GTiff",
        "height": height,
        "width": width,
        "transform": src.transform,
        "crs": src.crs,
    }

    with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rio.open(export_path, "w", **profile) as dst:
            if isinstance(array, list):
                stacked = np.stack(array)
                dst.write(stacked)
                del stacked
            else:
                dst.write(array)
            dst.descriptions = tuple(layer_names)
            if mask is not None:
                dst.write_mask((mask.astype(np.uint8) * 255))


def rasterize_vector(
    gdf: gpd.GeoDataFrame, reference_profile: dict, all_touched=True
) -> np.ndarray:
    """Rasterize a GeoDataFrame into a binary array using the reference rio profile."""
    height, width = reference_profile["height"], reference_profile["width"]
    pixel_size = reference_profile["transform"][0]
    out = np.zeros((height, width), dtype=rio.uint8)
    if len(gdf) == 0:
        return out

    # simplify geometries to the pixel size to improve computation time
    gdf_simple = gdf.simplify(tolerance=pixel_size, preserve_topology=True)

    # Vectorized geometry extraction
    shapes = list(((geom, 1) for geom in gdf_simple.geometry))

    # Use out parameter in rasterize
    features.rasterize(
        shapes=shapes,
        out=out,
        transform=reference_profile["transform"],
        all_touched=all_touched,
    )

    return out
