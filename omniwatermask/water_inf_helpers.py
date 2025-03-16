import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Union

import cv2
import numpy as np
import rasterio as rio
import torch
from omnicloudmask import predict_from_array
from scipy.optimize import minimize_scalar

from .target_builders import build_targets


def get_masked_iou(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weighted: bool = True,
) -> float:
    """
    Calculate IoU between source and target tensors with optional masking and weighting.

    Args:
        source: Binary tensor (0s and 1s)
        target: Binary tensor (0s and 1s) or weighted tensor (0, 1, 2, etc.) if weighted=True
        mask: Optional mask tensor (True values are excluded from calculation)
        weighted: If True, treats target as weighted values instead of binary

    Returns:
        float: IoU score (weighted if weighted=True)
    """
    if mask is not None:
        source = torch.logical_and(source, ~mask)
        target = torch.where(mask, torch.zeros_like(target), target)

    if weighted:
        # intersection = torch.minimum(source, target).sum().item()
        intersection = (target * source).sum().item()
        union = torch.maximum(source, target).sum().item()
    else:
        intersection = torch.logical_and(source, target).sum().item()
        union = torch.logical_or(source, target).sum().item()

    iou_score = intersection / union if union != 0 else 0
    return iou_score


def optimise_threshold(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    min_thresh: float = -0.3,
    max_thresh: float = 0.3,
    num_steps: int = 40,
) -> tuple[torch.Tensor, float]:
    """Get the optimal threshold to align the source tensor with the target tensor"""

    def objective(threshold):
        return -get_masked_iou(
            source=(source > threshold), target=target, mask=mask
        )  # Negative because we want to maximize

    result = minimize_scalar(
        objective,
        bounds=(min_thresh, max_thresh),
        method="bounded",
        options={"xatol": 0.0001, "maxiter": num_steps},
    )

    optimal_threshold = result.x  # type: ignore
    highest_accuracy = -result.fun  # type: ignore

    return source > optimal_threshold, float(highest_accuracy)


def get_intersection_ratio(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get the intersection ratio of each cluster in the source image with the target image
    returns a tensor of the same shape as the source image with the intersection ratios
    """
    source_np = source.numpy(force=True).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        source_np, connectivity=8
    )

    labeled_torch = torch.from_numpy(labels).to(source.device)

    intersection_ratios = torch.zeros_like(source, dtype=torch.float32)

    for label in range(1, num_labels):
        min_col, min_row, width, height, _ = stats[label]
        max_row, max_col = min_row + height, min_col + width

        cluster_mask_slice = (
            labeled_torch[min_row:max_row, min_col:max_col] == label
        ).float()
        pred_slice = target[min_row:max_row, min_col:max_col].float()

        variable_cluster_sum = cluster_mask_slice.sum()
        binary_cluster_sum = (cluster_mask_slice * pred_slice).sum()

        intersecting_ratio = binary_cluster_sum / variable_cluster_sum

        intersection_ratios[min_row:max_row, min_col:max_col] += (
            cluster_mask_slice * intersecting_ratio
        )

    return intersection_ratios


def optimise_by_threshold_and_overlap(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    scene_thresholds: tuple = (-0.3, 0.3),
    cluster_thresholds: tuple = (0.4, 0.6),
    scene_threshold_steps: int = 20,
    cluster_ratio_steps: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise the agreement of the source image with the target image by thresholding and overlapping"""
    thresholded_source, _ = optimise_threshold(
        source=source,
        target=target,
        mask=mask,
        min_thresh=scene_thresholds[0],
        max_thresh=scene_thresholds[1],
        num_steps=scene_threshold_steps,
    )

    cluster_with_intersection_ratios = get_intersection_ratio(
        source=thresholded_source, target=target
    )

    if mask is not None:
        cluster_with_intersection_ratios = cluster_with_intersection_ratios * ~mask

    cluster_filter_source, _ = optimise_threshold(
        source=cluster_with_intersection_ratios,
        target=target,
        mask=None,
        min_thresh=cluster_thresholds[0],
        max_thresh=cluster_thresholds[1],
        num_steps=cluster_ratio_steps,
    )
    return cluster_filter_source, source > thresholded_source


def optimise_patches(
    source: torch.Tensor,
    target: torch.Tensor,
    accuracy_tracker: torch.Tensor,
    cumulative_detections: torch.Tensor,
    patch_size: int,
    min_thresh: float,
    max_thresh: float,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise the agreement of the source image with the target image by thresholding and overlapping in patches"""
    max_height, max_width = source.shape

    for top in range(0, max_height, patch_size):
        bottom = min(top + patch_size, max_height)
        full_size_top = bottom - patch_size

        for left in range(0, max_width, patch_size):
            right = min(left + patch_size, max_width)
            full_size_left = right - patch_size

            target_patch = target[full_size_top:bottom, full_size_left:right]
            if mask is not None:
                mask_patch = mask[full_size_top:bottom, full_size_left:right]

            if target_patch.sum() != 0:
                source_patch = source[full_size_top:bottom, full_size_left:right]
                binary_source_patch, patch_accuracy = optimise_threshold(
                    source=source_patch,
                    target=target_patch,
                    mask=mask_patch if mask is not None else None,
                    min_thresh=min_thresh,
                    max_thresh=max_thresh,
                )
                cumulative_detections[top:bottom, left:right] += (
                    binary_source_patch[-(bottom - top) :, -(right - left) :].float()
                    * patch_accuracy
                )
                accuracy_tracker[top:bottom, left:right] += patch_accuracy

    return cumulative_detections, accuracy_tracker


def multi_scale_optimisation(
    source: torch.Tensor,
    target: torch.Tensor,
    patch_sizes: list[int],
    mask: Optional[torch.Tensor],
    min_thresh: float = -0.1,
    max_thresh: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimise the agreement of the source image with the target image by thresholding and overlapping at multiple scales,
    results of which are combined and further optimised to a binary output"""

    cumulative_detections, accuracy = optimise_threshold(
        source=source,
        target=target,
        min_thresh=min_thresh,
        max_thresh=max_thresh,
        mask=mask,
    )
    cumulative_detections = cumulative_detections.float() * accuracy
    accuracy_tracker = torch.zeros_like(source, dtype=torch.float) + accuracy

    for patch_size in patch_sizes:
        if patch_size < source.shape[0] and patch_size < source.shape[1]:
            cumulative_detections, accuracy_tracker = optimise_patches(
                target=target,
                source=source,
                accuracy_tracker=accuracy_tracker,
                cumulative_detections=cumulative_detections,
                patch_size=patch_size,
                min_thresh=min_thresh,
                max_thresh=max_thresh,
            )
    normalised_accuracy = cumulative_detections / accuracy_tracker

    if torch.isnan(normalised_accuracy).any():
        logging.debug("Normalised accuracy contains NaN values, setting to zeros")
        normalised_accuracy = torch.zeros_like(normalised_accuracy)

    threshold_and_cluster_optimised, threshold_optimised = (
        optimise_by_threshold_and_overlap(
            source=normalised_accuracy,
            target=target,
            mask=mask,
            scene_thresholds=(0, 1),
            scene_threshold_steps=10,
            cluster_ratio_steps=10,
        )
    )

    return (
        threshold_and_cluster_optimised,
        accuracy_tracker,
        cumulative_detections,
        threshold_optimised,
        normalised_accuracy,
    )


def get_NDWI(
    input_bands: np.ndarray, mosaic_device: Union[str, torch.device]
) -> torch.Tensor:
    input_bands_tensor = torch.from_numpy(input_bands.astype(np.float16)).to(
        mosaic_device
    )
    ndwi = (input_bands_tensor[1] - input_bands_tensor[3]) / (
        input_bands_tensor[1] + input_bands_tensor[3]
    )

    return ndwi


def make_composite_output(input_dict: dict) -> tuple[np.ndarray, list[str]]:
    output_layers = []
    layer_names = []
    # Get the shape of the first non-None layer
    for key, value in input_dict.items():
        if value is not None:
            shape = value.shape
            break
    for key, value in input_dict.items():
        #  if value is None set it to a zero tensor of the same shape, this avoids missing export layers
        if value is None:
            logging.info(f"Layer {key} is None, setting to zero tensor")
            value = torch.zeros(shape, dtype=torch.float32)
        output_layers.append(value.float().numpy(force=True).astype(np.float32))
        layer_names.append(key)
    output_layers = np.stack(output_layers)
    return output_layers, layer_names


def integrate_water_detection_methods(
    input_bands: np.ndarray,
    input_path: Path,
    cache_dir: Path,
    inference_dtype: torch.dtype,
    inference_device: torch.device,
    inference_patch_size: int,
    inference_overlap_size: int,
    batch_size: int,
    models: list[torch.nn.Module],
    use_cache: bool = True,
    patch_sizes: list[int] = [200, 400, 800, 1000],
    debug_output: bool = False,
    use_osm_water: bool = True,
    use_ndwi: bool = True,
    use_model: bool = True,
    use_osm_building_mask: bool = True,
    use_osm_roads_mask: bool = True,
    aux_vector_sources: list[Path] = [],
    aux_negative_vector_sources: list[Path] = [],
    mosaic_device: Union[str, torch.device] = "cpu",
    no_data_value: int = 0,
    optimise_model: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Combine the NDWI, model predictions and vector targets"""
    combined_water = []
    model_target = []
    ndwi_target = []
    negative_target = []
    logging.info("Integrating water detection methods")
    ndwi_conf_tensor = get_NDWI(input_bands=input_bands, mosaic_device=mosaic_device)

    # if zeros across all bands, set to no data
    no_data_mask = torch.tensor(np.all(input_bands == no_data_value, axis=0)).to(
        mosaic_device
    )
    no_data_mask = no_data_mask.to(inference_dtype)
    negative_target.append(no_data_mask)

    ndwi_conf_tensor = ndwi_conf_tensor.to(inference_dtype)

    logging.info("Building vector target in thread")
    vector_target_result_queue = Queue()
    vector_target_thread = Thread(
        target=build_targets,
        kwargs={
            "raster_src": rio.open(input_path),
            "osm_water": use_osm_water,
            "aux_vector_sources": aux_vector_sources,
            "device": mosaic_device,
            "cache_dir": cache_dir,
            "use_cache": use_cache,
            "queue": vector_target_result_queue,
        },
    )
    vector_target_thread.start()

    if use_osm_building_mask or use_osm_roads_mask:
        logging.info("Building negative targets in thread")
        negative_target_result_queue = Queue()
        negative_target_thread = Thread(
            target=build_targets,
            kwargs={
                "raster_src": rio.open(input_path),
                "osm_buildings": use_osm_building_mask,
                "osm_roads": use_osm_roads_mask,
                "aux_vector_sources": aux_negative_vector_sources,
                "device": mosaic_device,
                "cache_dir": cache_dir,
                "use_cache": use_cache,
                "queue": negative_target_result_queue,
            },
        )
        negative_target_thread.start()

    if use_model:
        logging.info("Predicting water mask using custom model")

        model_conf = predict_from_array(
            input_bands[:4],
            custom_models=models,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
            export_confidence=True,
            softmax_output=True,
            no_data_value=no_data_value,
            pred_classes=2,
            inference_device=inference_device,
            mosaic_device=mosaic_device,
            patch_size=inference_patch_size,
            patch_overlap=inference_overlap_size,
        )
        model_conf_tensor = torch.from_numpy(model_conf).to(mosaic_device)

        model_conf_tensor = model_conf_tensor.to(inference_dtype)

        model_conf_tensor = model_conf_tensor[1] - model_conf_tensor[0]

        model_binary = model_conf_tensor > 0.0

        ndwi_target.append(model_binary)
    else:
        model_conf_tensor = None
        model_binary = None

    if vector_target_thread.is_alive():
        logging.info("Waiting for vector targets to finish")
    vector_target_thread.join()
    vector_targets = vector_target_result_queue.get()

    if vector_targets is not None:
        model_target.append(vector_targets)
        ndwi_target.append(vector_targets)

    if use_osm_building_mask or use_osm_roads_mask:
        if negative_target_thread.is_alive():
            logging.info("Waiting for negative targets to finish")
        negative_target_thread.join()
        vector_negative_target = negative_target_result_queue.get()
        if vector_negative_target is not None:
            negative_target.append(vector_negative_target)

    if len(negative_target) > 0:
        negative_target = torch.stack(negative_target).sum(0) > 0
    else:
        negative_target = None

    if use_ndwi:
        logging.info("Optimising NDWI")
        if len(ndwi_target) > 0:
            ndwi_target = torch.stack(ndwi_target).sum(0)
        else:
            ndwi_target = torch.zeros_like(ndwi_conf_tensor, dtype=torch.bool)

        (
            NDWI_binary,
            NDWI_accuracy_tracker,
            NDWI_cumulative_detections,
            _,
            normalised_accuracy,
        ) = multi_scale_optimisation(
            source=ndwi_conf_tensor,
            target=ndwi_target,
            patch_sizes=patch_sizes,
            mask=negative_target,
        )
        logging.info("Multi-scale optimisation accuracy finished")
        combined_water.append(NDWI_binary)
        model_target.append(NDWI_binary)
        model_target.append(ndwi_conf_tensor > 0.5)

    else:
        NDWI_accuracy_tracker = None
        NDWI_cumulative_detections = None
        normalised_accuracy = None

    if len(model_target) > 0:
        model_target = torch.stack(model_target).sum(0)
    else:
        model_target = torch.zeros_like(ndwi_conf_tensor, dtype=torch.bool)

    if model_conf_tensor is not None:
        if optimise_model:
            logging.info("Optimising model predictions")
            model_binary_cleaned, _ = optimise_by_threshold_and_overlap(
                source=model_conf_tensor,
                target=model_target,
                mask=negative_target,
                scene_thresholds=(0, 1),
            )

            combined_water.append(model_binary_cleaned)
        else:
            logging.info("Using raw model predictions")
            combined_water.append(model_binary)
            model_binary_cleaned = None

    else:
        model_conf_tensor = None
        model_binary_cleaned = None

    combined_water = torch.stack(combined_water).sum(0) > 0

    if debug_output:
        logging.info("Exporting debug layers")
        final_output, layer_names = make_composite_output(
            {
                "Water predictions": combined_water,
                "NDWI binary": NDWI_binary,
                "NDWI target": ndwi_target,
                "NDWI raw": ndwi_conf_tensor,
                "NDWI cumulative detections": NDWI_cumulative_detections,
                "NDWI accuracy tracker": NDWI_accuracy_tracker,
                "NDWI normalised accuracy": normalised_accuracy,
                "Model binary cleaned": model_binary_cleaned,
                "Model binary": model_binary,
                "Model target": model_target,
                "Model confidence": model_conf_tensor,
                "Vector inputs": vector_targets,
                "Negative vector inputs": negative_target,
                "No data mask": no_data_mask,
            }
        )
    else:
        final_output = combined_water.numpy(force=True).astype(np.uint8)
        no_data_mask_np = (~(no_data_mask.bool())).numpy(force=True).astype(np.uint8)
        # final_output = np.expand_dims(final_output, axis=0)
        final_output = np.stack([final_output, no_data_mask_np])
        layer_names = ["Water predictions", "No data mask"]

    return final_output, layer_names
