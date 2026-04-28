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
        target: Binary tensor (0s and 1s) or weighted tensor
            (0, 1, 2, etc.) if weighted=True
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
    """Get the optimal threshold to align the source tensor with the target tensor."""
    device = source.device
    thresholds = torch.linspace(
        min_thresh, max_thresh, num_steps, dtype=source.dtype, device=device
    )

    # Keep best_iou and best_thresh as device tensors — avoids one GPU→CPU sync per
    # threshold step (80 stalls → 2 total when running on MPS/CUDA).
    best_iou = torch.tensor(-1.0, dtype=torch.float32, device=device)
    best_thresh = torch.tensor(float(min_thresh), dtype=source.dtype, device=device)

    # Mask out target pixels in-place to skip an extra full-tile alloc.
    target_f = target.float()
    mask_bool = mask.bool() if mask is not None else None
    if mask_bool is not None:
        target_f.masked_fill_(mask_bool, 0.0)

    for thresh in thresholds:
        # Keep binary as bool until the float promotion is unavoidable; saves
        # ~360 MB per iteration on a full S2 tile (480 MB float vs 120 MB bool).
        binary_b = source > thresh
        if mask_bool is not None:
            binary_b &= ~mask_bool
        binary = binary_b.float()
        del binary_b
        intersection = (target_f * binary).sum()
        union = torch.maximum(binary, target_f).sum()
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        improved = iou > best_iou
        best_iou = torch.where(improved, iou, best_iou)
        best_thresh = torch.where(improved, thresh, best_thresh)

    # Single sync per call to extract scalar results
    return source > best_thresh.item(), best_iou.item()


def get_intersection_ratio(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get the intersection ratio of each cluster in the source
    with the target. Returns a tensor of the same shape as the
    source image with the intersection ratios.
    """
    source_np = source.numpy(force=True).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        source_np, connectivity=8
    )

    if num_labels <= 1:
        return torch.zeros_like(source, dtype=torch.float32)

    labeled = torch.from_numpy(labels).to(source.device)
    lflat = labeled.view(-1).long()
    tflat = target.view(-1).float()
    ones = torch.ones(lflat.shape[0], dtype=torch.float32, device=source.device)

    # Per-label pixel count and target overlap — O(H×W) scatter, no Python loop
    comp_size = torch.zeros(num_labels, dtype=torch.float32, device=source.device)
    comp_size.scatter_add_(0, lflat, ones)

    tgt_sum = torch.zeros(num_labels, dtype=torch.float32, device=source.device)
    tgt_sum.scatter_add_(0, lflat, tflat)

    ratios = tgt_sum / comp_size.clamp(min=1)
    ratios[0] = 0.0  # background label

    return ratios[lflat].view(source.shape)


def optimise_by_threshold_and_overlap(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    scene_thresholds: tuple = (-0.3, 0.3),
    cluster_thresholds: tuple = (0.4, 0.6),
    scene_threshold_steps: int = 20,
    cluster_ratio_steps: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise source-target agreement by thresholding and overlapping."""
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
    _num_steps: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise source-target agreement by thresholding in patches.

    Uses fully vectorized (K × M × P × P) threshold search: all candidate
    thresholds and all patches in a mini-batch are evaluated in a single
    tensor broadcast, eliminating all Python-level threshold iteration.
    """
    max_height, max_width = source.shape
    device = source.device

    src_patches: list[torch.Tensor] = []
    tgt_patches: list[torch.Tensor] = []
    msk_patches: list[torch.Tensor] = []
    coords: list[tuple[int, int, int, int]] = []

    # Check empty patches on CPU to avoid one MPS sync per patch (thousands of stalls).
    target_cpu = target.cpu() if target.device.type != "cpu" else target

    for top in range(0, max_height, patch_size):
        bottom = min(top + patch_size, max_height)
        fs_top = bottom - patch_size
        for left in range(0, max_width, patch_size):
            right = min(left + patch_size, max_width)
            fs_left = right - patch_size
            if target_cpu[fs_top:bottom, fs_left:right].sum() == 0:
                continue
            src_patches.append(source[fs_top:bottom, fs_left:right])
            tgt_patches.append(target[fs_top:bottom, fs_left:right])
            if mask is not None:
                msk_patches.append(mask[fs_top:bottom, fs_left:right])
            coords.append((top, bottom, left, right))

    if not src_patches:
        return cumulative_detections, accuracy_tracker

    thresholds = torch.linspace(
        min_thresh, max_thresh, _num_steps, dtype=source.dtype, device=device
    )

    # Batch size: keep (K, M, P, P) float32 under ~256 MB.
    # K * M * P^2 * 4 bytes < 256e6  →  M < 256e6 / (K * P^2 * 4)
    _bytes_per_kpp = _num_steps * patch_size * patch_size * 4
    batch_size = max(1, 256_000_000 // _bytes_per_kpp)

    for b in range(0, len(src_patches), batch_size):
        end = min(b + batch_size, len(src_patches))

        src_b = torch.stack(src_patches[b:end])  # (M, P, P)
        tgt_b = torch.stack(tgt_patches[b:end]).float()  # (M, P, P)
        msk_b = torch.stack(msk_patches[b:end]) if mask is not None else None

        tgt_m = tgt_b * (~msk_b).float() if msk_b is not None else tgt_b  # (M, P, P)

        # Broadcast all K thresholds against all M patches simultaneously.
        # binary_all: (K, M, P, P) bool — one tensor op replaces K Python iterations.
        binary_all = src_b.unsqueeze(0) > thresholds.view(-1, 1, 1, 1)
        binary_f = binary_all.float()
        if msk_b is not None:
            binary_f = binary_f * (~msk_b).unsqueeze(0).float()

        tgt_e = tgt_m.unsqueeze(0)  # (1, M, P, P)
        inter = (tgt_e * binary_f).sum(dim=(-2, -1))  # (K, M)
        union = torch.maximum(binary_f, tgt_e).sum(dim=(-2, -1))  # (K, M)
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))  # (K, M)

        best_k = iou.argmax(dim=0)  # (M,)
        best_iou = iou.max(dim=0).values  # (M,)
        best_thresh_vals = thresholds[best_k]  # (M,)

        del binary_all, binary_f, inter, union, iou, tgt_e

        # Two syncs per batch (.tolist()) rather than two per patch (.item() × M).
        pa_list = best_iou.tolist()
        thresh_list = best_thresh_vals.tolist()

        for i, (top, bottom, left, right) in enumerate(coords[b:end]):
            pa = pa_list[i]
            if pa == 0:
                continue
            ph, pw = bottom - top, right - left
            binary = (src_patches[b + i] > thresh_list[i]).float()
            cumulative_detections[top:bottom, left:right] += binary[-ph:, -pw:] * pa
            accuracy_tracker[top:bottom, left:right] += pa

    return cumulative_detections, accuracy_tracker


def multi_scale_optimisation(
    source: torch.Tensor,
    target: torch.Tensor,
    patch_sizes: list[int],
    mask: Optional[torch.Tensor],
    min_thresh: float = -0.1,
    max_thresh: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimise source-target agreement by thresholding at multiple
    scales, combining results and further optimising to binary."""

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
                mask=mask,
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


def make_composite_output(
    input_dict: dict,
) -> tuple[list[Union[torch.Tensor, np.ndarray, None]], list[str]]:
    """Return debug layers for export.

    Layers are kept in their native form (torch tensor / numpy / None) and only
    converted to float32 at write time, so we never materialise all 14 layers
    as float32 numpy in memory at once. None placeholders are converted to
    zeros lazily during export.
    """
    output_layers: list[Union[torch.Tensor, np.ndarray, None]] = []
    layer_names: list[str] = []
    for key, value in input_dict.items():
        if value is None:
            logging.info(f"Layer {key} is None, will be filled with zeros at export")
        output_layers.append(value)
        layer_names.append(key)
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
    patch_sizes: Optional[list[int]] = None,
    debug_output: bool = False,
    use_osm_water: bool = True,
    use_ndwi: bool = True,
    use_model: bool = True,
    use_osm_building_mask: bool = True,
    use_osm_roads_mask: bool = True,
    aux_vector_sources: Optional[list[Path]] = None,
    aux_negative_vector_sources: Optional[list[Path]] = None,
    mosaic_device: Union[str, torch.device] = "cpu",
    no_data_value: int = 0,
    optimise_model: bool = True,
) -> tuple[
    np.ndarray | list[Union[torch.Tensor, np.ndarray, None]],
    list[str],
    np.ndarray,
]:
    """Combine the NDWI, model predictions and vector targets"""
    if patch_sizes is None:
        patch_sizes = [200, 400, 800, 1000]
    if aux_vector_sources is None:
        aux_vector_sources = []
    if aux_negative_vector_sources is None:
        aux_negative_vector_sources = []
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
        del model_conf

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
        # Iterative OR avoids the (N, H, W) stacked allocation that
        # torch.stack(...).sum(0) > 0 materialises (~N×120 MB on a full S2 tile).
        neg_iter = iter(negative_target)
        negative_combined = next(neg_iter).bool().clone()
        for t in neg_iter:
            negative_combined |= t.bool()
        negative_target = negative_combined
        del negative_combined
    else:
        negative_target = None

    if use_ndwi:
        logging.info("Optimising NDWI")
        if len(ndwi_target) > 0:
            # Accumulate weighted target as uint8 in-place — saves ~1 GB peak vs
            # torch.stack(...).sum(0), which builds an (N,H,W) bool stack and
            # then promotes the result to int64.
            it = iter(ndwi_target)
            ndwi_target = next(it).to(torch.uint8).clone()
            for t in it:
                ndwi_target += t.to(torch.uint8)
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
        NDWI_binary = None
        ndwi_target = None
        NDWI_accuracy_tracker = None
        NDWI_cumulative_detections = None
        normalised_accuracy = None

    if len(model_target) > 0:
        it = iter(model_target)
        model_target = next(it).to(torch.uint8).clone()
        for t in it:
            model_target += t.to(torch.uint8)
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

    # Final fusion: OR all binary water predictions together. Iterative OR
    # avoids stacking (~N×120 MB) and the int64 sum allocation (~960 MB).
    cw_iter = iter(combined_water)
    combined_water = next(cw_iter).bool().clone()
    for t in cw_iter:
        combined_water |= t.bool()
    del cw_iter

    valid_mask = (~(no_data_mask.bool())).numpy(force=True).astype(np.uint8)

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
        final_output = np.expand_dims(final_output, axis=0)
        layer_names = ["Water predictions"]

    return final_output, layer_names, valid_mask
