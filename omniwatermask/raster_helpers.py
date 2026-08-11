from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from numpy.typing import NDArray
from rasterio import features
from rasterio.transform import from_bounds


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

        left, bottom, right, top = src.bounds
        profile = src.profile.copy()
        profile.update(
            height=new_height,
            width=new_width,
            transform=from_bounds(left, bottom, right, top, new_width, new_height),
            alpha="unspecified",
        )
        data = src.read(out_shape=(src.count, new_height, new_width))

        with rio.open(resample_path, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = src.descriptions
            dst.colorinterp = src.colorinterp

    return resample_path


def export_to_disk(
    array: NDArray[Any],
    export_path: Path,
    source_path: Path,
    layer_names: list[str],
    nodata_mask: Optional[NDArray[Any]] = None,
) -> None:
    """Export the array to disk as a GeoTIFF.

    If ``nodata_mask`` is provided (1 = valid, 0 = no data) it is written as a
    GDAL dataset mask via ``dst.write_mask`` rather than as a regular band, so
    GIS software (e.g. QGIS) treats no-data pixels as transparent. The mask is
    embedded inside the GeoTIFF (``GDAL_TIFF_INTERNAL_MASK``) rather than written
    as a separate ``.tif.msk`` sidecar, so it travels with the file.
    """
    with rio.open(source_path) as src:
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

    with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rio.open(export_path, "w", **profile) as dst:
            dst.write(array)
            dst.descriptions = layer_names
            if nodata_mask is not None:
                dst.write_mask((nodata_mask * 255).astype("uint8"))


def rasterize_vector(
    gdf: gpd.GeoDataFrame, reference_profile: dict[str, Any], all_touched: bool = False
) -> NDArray[Any]:
    """Rasterize a GeoDataFrame into a binary array using the reference rio profile.

    With ``all_touched=False`` (the default) a pixel is only set when its centre
    falls inside a geometry, so a feature must cover roughly the majority of a
    pixel for it to flip on. This avoids small sub-pixel features (e.g. a tiny
    pond inside a single 10 m Sentinel-2 pixel) turning on the whole pixel, which
    is what ``all_touched=True`` would do for any feature that merely clips it.
    """
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
