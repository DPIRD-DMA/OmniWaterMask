from pathlib import Path
from typing import Union

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
    array: np.ndarray, export_path: Path, source_path: Path, layer_names: list[str]
):
    """Export the array to disk as a GeoTIFF"""
    src = rio.open(source_path)
    profile = {
        "dtype": array.dtype,
        "count": array.shape[0],
        "compress": "lzw",
        "nodata": None,
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "transform": src.transform,
        "crs": src.crs,
    }

    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = layer_names


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
    gdf = gdf.simplify(tolerance=pixel_size, preserve_topology=True)

    # Vectorized geometry extraction
    shapes = list(((geom, 1) for geom in gdf.geometry))

    # Use out parameter in rasterize
    features.rasterize(
        shapes=shapes,
        out=out,
        transform=reference_profile["transform"],
        all_touched=all_touched,
    )

    return out
