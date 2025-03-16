import logging
from pathlib import Path
from typing import Optional, Union

import rasterio as rio
import torch
from omnicloudmask.model_utils import (
    default_device,
    get_torch_dtype,
    load_model,
    load_model_from_weights,
)
from tqdm.auto import tqdm

from .__version__ import __version__
from .download_models import get_models
from .raster_helpers import export_to_disk, resample_input
from .vector_cache import initialize_db
from .water_inf_helpers import integrate_water_detection_methods


def collect_models(
    model_path: list[str] | list[Path] | str | Path,
    inference_device: torch.device,
    inference_dtype: torch.dtype,
) -> list[torch.nn.Module]:
    models = []
    if model_path != "":
        if not isinstance(model_path, list):
            model_path_list = [model_path]
        else:
            model_path_list = model_path

        for model_p in model_path_list:
            model = load_model(
                model_path=model_p,
                device=inference_device,
                dtype=inference_dtype,
            )
            models.append(model)
    # if no model path is provided, use the default model
    else:
        for model_details in get_models():
            models.append(
                load_model_from_weights(
                    model_name=model_details["timm_model_name"],
                    weights_path=model_details["Path"],
                    device=inference_device,
                    dtype=inference_dtype,
                    in_chans=4,
                    n_out=2,
                )
            )
    return models


def make_water_mask(
    scene_paths: list[Path] | list[str] | Path | str,
    band_order: list[int],
    batch_size: int = 1,
    version: Union[str, int, float] = f"OmniWaterMask_{__version__}",
    output_dir: Optional[Path] = None,
    mosaic_device: Union[str, torch.device] = default_device(),
    inference_device: Union[str, torch.device] = default_device(),
    aux_vector_sources: list[Path] = [],
    aux_negative_vector_sources: list[Path] = [],
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    no_data_value: int = 0,
    inference_patch_size: int = 1000,
    inference_overlap_size: int = 300,
    overwrite: bool = True,
    use_cache: bool = True,
    use_osm_water: bool = True,
    use_osm_building: bool = True,
    use_osm_roads: bool = True,
    cache_dir: Path = Path.cwd() / "OWM_cache",
) -> list[Path]:
    return make_water_mask_debug(
        scene_paths=scene_paths,
        band_order=band_order,
        batch_size=batch_size,
        version=version,
        output_dir=output_dir,
        mosaic_device=mosaic_device,
        use_osm_water=use_osm_water,
        use_osm_building=use_osm_building,
        use_osm_roads=use_osm_roads,
        aux_vector_sources=aux_vector_sources,
        aux_negative_vector_sources=aux_negative_vector_sources,
        inference_dtype=inference_dtype,
        inference_device=inference_device,
        inference_patch_size=inference_patch_size,
        inference_overlap_size=inference_overlap_size,
        no_data_value=no_data_value,
        overwrite=overwrite,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )


def make_water_mask_debug(
    scene_paths: list[Path] | list[str] | Path | str,
    band_order: list[int],
    batch_size: int = 1,
    version: Union[str, int, float] = f"OmniWaterMask_{__version__}",
    model_path: list[str] | list[Path] | str | Path = "",
    output_dir: Optional[Path] = None,
    debug_output: bool = False,
    mosaic_device: Union[str, torch.device] = default_device(),
    use_osm_water: bool = True,
    use_model: bool = True,
    use_ndwi: bool = True,
    use_osm_building: bool = True,
    use_osm_roads: bool = True,
    aux_vector_sources: list[Path] = [],
    aux_negative_vector_sources: list[Path] = [],
    resample_res: Optional[Union[int, float]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    inference_device: Union[str, torch.device] = default_device(),
    inference_patch_size: int = 1000,
    inference_overlap_size: int = 300,
    no_data_value: int = 0,
    overwrite: bool = True,
    use_cache: bool = True,
    optimise_model: bool = True,
    cache_dir: Path = Path.cwd() / "water_vectors_cache",
) -> list[Path]:
    # Make sure that the correct options are set
    if not use_osm_water or aux_vector_sources:
        if not use_model:
            raise ValueError(
                "If not using vector targets (OSM or aux_vector_sources), you must enable use_model"
            )
        if not use_ndwi:
            raise ValueError(
                "If not using vector targets (OSM or aux_vector_sources), you must enable use_ndwi"
            )

    if use_cache:
        initialize_db(cache_dir)

    if isinstance(scene_paths, (str, Path)):
        scene_paths_list = [Path(scene_paths)]
    elif isinstance(scene_paths, list):
        scene_paths_list = [Path(image) for image in scene_paths]
    else:
        raise ValueError(
            "scene_paths must be a list of Paths (or strings) or a path (or string)"
        )
    #  Convert the inference device and dtype to torch types
    inference_dtype_torch = get_torch_dtype(inference_dtype)
    inference_device_torch = torch.device(inference_device)

    # Load the models
    models = collect_models(
        model_path=model_path,
        inference_device=inference_device_torch,
        inference_dtype=inference_dtype_torch,
    )

    # Set up the progress bar
    p_bar = tqdm(
        total=len(scene_paths_list), desc=f"Using {inference_device}", leave=False
    )

    output_paths = []

    for input_image in scene_paths_list:
        # if no output directory is provided, save the output in the same directory as the input image
        if output_dir is None:
            output_dir_set = input_image.parent
        else:
            output_dir_set = output_dir

        output_dir_set.mkdir(exist_ok=True, parents=True)
        # Optionally resample the input image
        if resample_res is not None:
            logging.info(f"Resampling {input_image.name} to {resample_res}m")
            input_image = resample_input(
                input_path=input_image,
                resample_res=resample_res,
                output_dir=output_dir_set,
            )

        debug_str = "_debug" if debug_output else ""
        export_path = output_dir_set / (input_image.stem + f"_{version}{debug_str}.tif")
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
            inference_dtype=inference_dtype_torch,
            inference_device=inference_device_torch,
            inference_patch_size=inference_patch_size,
            inference_overlap_size=inference_overlap_size,
            batch_size=batch_size,
            use_osm_water=use_osm_water,
            aux_vector_sources=aux_vector_sources,
            aux_negative_vector_sources=aux_negative_vector_sources,
            mosaic_device=mosaic_device,
            use_ndwi=use_ndwi,
            use_model=use_model,
            use_osm_building_mask=use_osm_building,
            use_osm_roads_mask=use_osm_roads,
            cache_dir=cache_dir,
            use_cache=use_cache,
            optimise_model=optimise_model,
            no_data_value=no_data_value,
            models=models,
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
