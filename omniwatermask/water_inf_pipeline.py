import logging
from pathlib import Path
from typing import Optional, Union

import rasterio as rio
import torch
from tqdm.auto import tqdm

from .__version__ import __version__
from .raster_helpers import export_to_disk, resample_input
from .water_inf_helpers import integrate_water_detection_methods


def extract_water(
    input_images: list[Path] | list[str] | Path | str,
    band_order: list[int],
    batch_size: int = 1,
    version: Union[str, int, float] = f"OmniWaterMask_{__version__}",
    model_path: list[str] | list[Path] | str | Path = "",
    output_dir: Optional[Path] = None,
    debug_output: bool = False,
    combine_device: str = "cpu",
    use_osm: bool = True,
    use_model: bool = True,
    use_ndwi: bool = True,
    use_osm_building: bool = True,
    use_osm_roads: bool = False,
    aux_vector_sources: list[Path] = [],
    aux_negative_vector_sources: list[Path] = [],
    resample_res: Optional[Union[int, float]] = None,
    inference_dtype: str = "bf16",
    inference_device: str = "cuda",
    inference_patch_size: int = 1000,
    inference_overlap_size: int = 300,
    overwrite: bool = False,
    use_cache: bool = True,
    optimise_model: bool = True,
    cache_dir: Path = Path.cwd() / "water_vectors_cache",
    regression_model: bool = False,
) -> list[Path]:
    vector_priors = use_osm or aux_vector_sources
    if not vector_priors:
        if not use_model:
            raise ValueError(
                "If not using vector priors (OSM or aux_vector_sources), you must enable use_model"
            )
        if not use_ndwi:
            raise ValueError(
                "If not using vector priors (OSM or aux_vector_sources), you must enable use_ndwi"
            )
    if isinstance(input_images, (str, Path)):
        input_images_list = [Path(input_images)]

    p_bar = tqdm(total=len(input_images_list))
    output_paths = []

    for input_image in input_images_list:
        if output_dir is None:
            output_dir_set = input_image.parent
        else:
            output_dir_set = output_dir
        output_dir_set.mkdir(exist_ok=True, parents=True)

        if resample_res is not None:
            logging.info(f"Resampling {input_image.name} to {resample_res}m")
            input_image = resample_input(
                input_path=input_image,
                resample_res=resample_res,
                output_dir=output_dir_set,
            )

        debug_str = "_debug" if debug_output else ""
        export_path = output_dir_set / (input_image.stem + f"_{version}{debug_str}.tif")
        logging.info(f"Exporting to {export_path}")
        output_paths.append(export_path)

        if export_path.exists() and not overwrite:
            logging.info(f"Skipping {input_image.name} as it already exists")
            p_bar.update(1)
            p_bar.refresh()
            continue

        logging.info(f"Processing {input_image.name}")
        input_src = rio.open(input_image)
        input_bands = input_src.read(band_order)

        logging.info(f"Predicting water mask for {input_image.name}")
        water_predictions, layer_names = integrate_water_detection_methods(
            input_bands=input_bands,
            input_path=input_image,
            debug_output=debug_output,
            model_path=model_path,
            inference_dtype=inference_dtype,
            inference_device=inference_device,
            inference_patch_size=inference_patch_size,
            inference_overlap_size=inference_overlap_size,
            batch_size=batch_size,
            use_osm=use_osm,
            aux_vector_sources=aux_vector_sources,
            aux_negative_vector_sources=aux_negative_vector_sources,
            combine_device=combine_device,
            use_ndwi=use_ndwi,
            use_model=use_model,
            use_osm_building_mask=use_osm_building,
            use_osm_roads_mask=use_osm_roads,
            cache_dir=cache_dir,
            use_cache=use_cache,
            optimise_model=optimise_model,
            regression_model=regression_model,
        )
        logging.info(f"Exporting {input_image.name} to {export_path}")
        export_to_disk(
            array=water_predictions,
            export_path=export_path,
            source_path=input_image,
            layer_names=layer_names,
        )
        p_bar.update(1)

    p_bar.refresh()
    p_bar.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_paths
