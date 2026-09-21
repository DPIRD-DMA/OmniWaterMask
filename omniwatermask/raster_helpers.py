from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.shutil
import torch
from numpy.typing import NDArray
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT


def resample_input(input_path: Path, resample_res: Union[int, float]) -> str:
    """Describe ``input_path`` at ``resample_res`` without materialising it.

    Returns a ``/vsimem`` path to a VRT - a couple of KB of XML saying "I am a
    raster of this size on this grid; fetch my pixels from that GeoTIFF". It
    opens like any other raster, so every consumer downstream still just takes a
    path, but the resampled scene only ever exists as the array whoever reads it
    asks for.

    This used to write a full resampled GeoTIFF next to the output: a complete
    second copy of every band on disk, including the bands the caller never
    reads, which was then decoded a second time on the way back in.

    The path is only valid within this run, and only while the VRT lives - it
    records an absolute path to its source, and nothing cleans up ``/vsimem``
    for you. Callers own it, and must ``rasterio.shutil.delete`` it when done.
    """
    vrt_path = f"/vsimem/owm_resample_{uuid4().hex}.vrt"
    # The VRT resolves its source when reopened, by which point this dataset is
    # long closed and the working directory is no longer anyone's promise.
    source = Path(input_path).resolve()

    with rio.open(source) as src:
        scale_factor = src.res[0] / resample_res
        new_height = round(src.height * scale_factor)
        new_width = round(src.width * scale_factor)

        left, bottom, right, top = src.bounds
        with WarpedVRT(
            src,
            crs=src.crs,
            transform=from_bounds(left, bottom, right, top, new_width, new_height),
            width=new_width,
            height=new_height,
            # Matches what the read-and-write version did: rasterio's default
            # for a decimated read is nearest, so the pixels are unchanged.
            resampling=Resampling.nearest,
        ) as vrt:
            rasterio.shutil.copy(vrt, vrt_path, driver="VRT")

    return vrt_path


def export_to_disk(
    array: Union[NDArray[Any], list[Optional["torch.Tensor"]]],
    export_path: Path,
    source_path: Union[str, Path],
    layer_names: list[str],
    nodata_mask: Optional[NDArray[Any]] = None,
) -> None:
    """Export the array to disk as a GeoTIFF.

    ``array`` is either a stacked numpy array - the single-band water mask - or
    a list of debug layers, which are written one band at a time. The list form
    exists to keep peak memory down: the debug output is 14 layers, and holding
    them all as float32 alongside a stacked copy costs 12.6 GB on a full
    Sentinel-2 tile, to write bands that are serialised individually anyway.
    Each layer is promoted as it is written and released immediately after, so
    only one float32 band is live at a time.

    If ``nodata_mask`` is provided (1 = valid, 0 = no data) it is written as a
    GDAL dataset mask via ``dst.write_mask`` rather than as a regular band, so
    GIS software (e.g. QGIS) treats no-data pixels as transparent. The mask is
    embedded inside the GeoTIFF (``GDAL_TIFF_INTERNAL_MASK``) rather than written
    as a separate ``.tif.msk`` sidecar, so it travels with the file.
    """
    layers: Optional[list[Optional["torch.Tensor"]]] = (
        array if isinstance(array, list) else None
    )
    if layers is not None:
        shape = _first_layer_shape(layers)
        count, height, width = len(layers), shape[-2], shape[-1]
        dtype: Any = "float32"
    else:
        assert not isinstance(array, list)
        count, height, width = array.shape[0], array.shape[1], array.shape[2]
        dtype = array.dtype

    with rio.open(source_path) as src:
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
    if layers is not None:
        # Writing band by band only lays out and compresses well with BAND
        # interleave and tiles; under the default PIXEL interleave LZW has to
        # decode and re-encode every strip once per band.
        profile.update(interleave="band", tiled=True, blockxsize=512, blockysize=512)

    with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rio.open(export_path, "w", **profile) as dst:
            if layers is not None:
                for index in range(count):
                    band = _layer_as_float32(layers[index], height, width)
                    # Drop the caller's reference as well as ours, so a layer
                    # nothing else holds is freed before the next is promoted.
                    layers[index] = None
                    dst.write(band, index + 1)
                    del band
            else:
                dst.write(array)
            dst.descriptions = layer_names
            if nodata_mask is not None:
                dst.write_mask((nodata_mask * 255).astype("uint8"))


def _first_layer_shape(layers: list[Optional["torch.Tensor"]]) -> tuple[int, ...]:
    for layer in layers:
        if layer is not None:
            return tuple(layer.shape)
    raise ValueError("export_to_disk was given no non-None layers")


def _layer_as_float32(
    layer: Optional["torch.Tensor"], height: int, width: int
) -> NDArray[Any]:
    """Promote one debug layer to the float32 band the GeoTIFF stores."""
    if layer is None:
        return np.zeros((height, width), dtype=np.float32)
    return layer.float().numpy(force=True).astype(np.float32, copy=False)


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
