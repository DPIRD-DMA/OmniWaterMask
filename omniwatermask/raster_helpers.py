from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
import torch
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


def _layer_to_float32_band(
    value: Any, height: int, width: int
) -> np.ndarray:
    """Convert a single debug layer to a 2-D float32 numpy array.

    Accepts torch tensors, numpy arrays, or None (rendered as zeros).
    """
    if value is None:
        return np.zeros((height, width), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.float().numpy(force=True).astype(np.float32, copy=False)
    if isinstance(value, np.ndarray):
        if value.dtype == np.float32:
            return value
        return value.astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def export_to_disk(
    array: Union[np.ndarray, list[Any]],
    export_path: Path,
    source_path: Path,
    layer_names: list[str],
    mask: Optional[np.ndarray] = None,
):
    """Export the array to disk as a GeoTIFF.

    For the debug path, ``array`` may be a list of mixed torch tensors / numpy
    arrays / ``None``. Each band is converted to float32 and written one at a
    time, then released — this avoids the ~12 GB peak from holding 14 float32
    arrays plus a stacked copy in memory.

    If ``mask`` is provided (2-D, 1=valid, 0=nodata), it is written as a GDAL
    internal mask band so QGIS/GDAL render nodata pixels as transparent.
    """
    src = rio.open(source_path)

    is_list = isinstance(array, list)
    if is_list:
        count = len(array)
        # Find shape from first non-None layer; debug layers are always written
        # as float32 to keep all bands a single dtype within the GeoTIFF.
        ref = next((v for v in array if v is not None), None)
        if ref is None:
            raise ValueError("export_to_disk: layer list is empty / all None")
        height, width = ref.shape[-2], ref.shape[-1]
        dtype = "float32"
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
    # For multi-band debug output we write band-by-band, which only produces
    # well-compressed, correctly-laid-out output when the GeoTIFF uses BAND
    # interleave + tiles. With the default PIXEL interleave LZW would have to
    # decode/encode each strip per band and the output ends up ~6× larger.
    if is_list:
        profile["interleave"] = "band"
        profile["tiled"] = True
        profile["blockxsize"] = 512
        profile["blockysize"] = 512

    with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rio.open(export_path, "w", **profile) as dst:
            if is_list:
                # Pop-then-write so source torch tensors / numpy arrays drop
                # their refcount as soon as we've serialised them. Peak adds
                # ~480 MB for one float32 conversion at a time instead of
                # ~13 GB for the previous list+stack pattern.
                for i in range(count):
                    band = _layer_to_float32_band(array[i], height, width)
                    array[i] = None
                    dst.write(band, i + 1)
                    del band
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
