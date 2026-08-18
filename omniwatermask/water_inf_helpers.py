import math
import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional, Union

import cv2
import numpy as np
import rasterio as rio
import torch
from numpy.typing import NDArray
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
    """Get the optimal threshold to align the source tensor with the target tensor"""

    # Only ``source > threshold`` varies across the minimize_scalar evaluations,
    # so the mask-dependent work (inverting the mask and zeroing the masked
    # target) is hoisted out of the objective and computed once here instead of
    # on every evaluation. Mirrors get_masked_iou's weighted=True branch.
    if mask is not None:
        inv_mask = ~mask
        masked_target = torch.where(mask, torch.zeros_like(target), target)
    else:
        inv_mask = None
        masked_target = target

    def objective(threshold: float) -> float:
        source_bin = source > threshold
        if inv_mask is not None:
            source_bin = torch.logical_and(source_bin, inv_mask)
        intersection = masked_target * source_bin
        union = torch.maximum(source_bin, masked_target)
        # Stack the two reductions so the device->host sync happens once.
        intersection_sum, union_sum = torch.stack(
            [intersection.sum(), union.sum()]
        ).tolist()
        iou_score = intersection_sum / union_sum if union_sum != 0 else 0
        return -iou_score  # Negative because we want to maximize

    result = minimize_scalar(
        objective,
        bounds=(min_thresh, max_thresh),
        method="bounded",
        options={"xatol": 0.0001, "maxiter": num_steps},
    )

    optimal_threshold = result.x
    highest_accuracy = -result.fun

    return source > optimal_threshold, float(highest_accuracy)


def _exact_histogram(
    bucket: torch.Tensor, weights: torch.Tensor, num_bins: int, max_weight: float
) -> NDArray[Any]:
    """Sum ``weights`` into ``num_bins`` bins without losing counts to rounding.

    A float32 accumulator saturates at 2**24: once a bin reaches 16,777,216,
    adding 1.0 rounds straight back to itself. A full Sentinel-2 tile is 120.5M
    pixels and most of them land in a handful of bins, so a plain float32
    histogram silently stops counting and the IoU curve it feeds becomes
    meaningless - it put 0.9% of a 74%-water scene under water. Integer
    accumulation would be exact but scatter_add on int32 or int64 runs ~100x
    slower on MPS (4.8s against 48ms for the same reduction), and float64 is
    unavailable there.

    Where the total cannot reach 2**24 the plain reduction is already exact and
    is used as-is. Only above that is the work split into chunks small enough
    that no accumulator can saturate, with the resulting (chunks, bins) table
    reduced in float64 on the CPU. Patches take the first path and full scenes
    the second, so the extra work lands only on the calls that need it.
    """
    n = bucket.numel()
    headroom = max(1.0, math.ceil(max_weight))

    if n * headroom <= (1 << 24):
        table = torch.zeros(num_bins, device=bucket.device).scatter_add_(
            0, bucket, weights
        )
        return table.cpu().double().numpy()

    chunk = max(1, int((1 << 24) // headroom))
    pad = (-n) % chunk
    if pad:
        bucket = torch.nn.functional.pad(bucket, (0, pad))
        weights = torch.nn.functional.pad(weights, (0, pad))
    n_chunks = (n + pad) // chunk

    table = torch.zeros(n_chunks, num_bins, device=bucket.device).scatter_add_(
        1, bucket.view(n_chunks, chunk), weights.view(n_chunks, chunk)
    )
    return table.cpu().double().numpy().sum(0)


def get_intersection_ratio(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get the intersection ratio of each cluster in the source
    with the target. Returns a tensor of the same shape as the
    source image with the intersection ratios.
    """
    source_np = source.numpy(force=True).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        source_np, connectivity=8
    )

    labeled_torch = torch.from_numpy(labels).to(source.device).long()

    # Every cluster's ratio is (sum of target over its pixels) / (its pixel
    # count), so one scatter_add over the label image replaces the per-label
    # Python loop this used to run - that sliced each cluster's bounding box
    # and reduced it separately, paying a kernel launch per cluster.
    # torch.bincount would express the same thing more directly but is
    # pathologically slow on MPS (~100x slower than the loop it replaces),
    # so the reduction is written as a scatter_add instead.
    # Accumulated through _exact_histogram rather than a bare scatter_add: a
    # single component can be enormous - the ocean in a coastal Sentinel-2 tile
    # is one blob of ~89M pixels - and a float32 accumulator stops counting at
    # 2**24. That understated such a cluster's ratio as ~0.19 instead of ~1.0,
    # putting it under the cluster filter's threshold and deleting the ocean
    # from the mask. The per-component torch.sum this replaced was a pairwise
    # reduction and did not have that problem.
    flat_target = target.reshape(-1).float()
    max_weight = float(flat_target.max().item()) if flat_target.numel() else 1.0
    cluster_sums = _exact_histogram(
        labeled_torch.reshape(-1), flat_target, num_labels, max_weight
    )

    # cv2 already counts each component's pixels, so the denominator is read
    # off the stats table rather than reduced a second time. The division stays
    # in float64 on the host, where the large cluster sums are exact, and only
    # the resulting ratios - all small - are carried back to the device.
    cluster_sizes = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    ratios_np = cluster_sums / np.maximum(cluster_sizes, 1.0)
    # the background label is not a cluster
    ratios_np[0] = 0.0

    ratios = torch.from_numpy(ratios_np.astype(np.float32)).to(source.device)
    return ratios[labeled_torch]


def optimise_by_threshold_and_overlap(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    scene_thresholds: tuple[float, float] = (-0.3, 0.3),
    cluster_thresholds: tuple[float, float] = (0.4, 0.6),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise source-target agreement by thresholding in patches."""
    max_height, max_width = source.shape

    for top in range(0, max_height, patch_size):
        bottom = min(top + patch_size, max_height)
        full_size_top = bottom - patch_size

        for left in range(0, max_width, patch_size):
            right = min(left + patch_size, max_width)
            full_size_left = right - patch_size

            target_patch = target[full_size_top:bottom, full_size_left:right]
            mask_patch = None
            if mask is not None:
                mask_patch = mask[full_size_top:bottom, full_size_left:right]

            if target_patch.sum() != 0:
                source_patch = source[full_size_top:bottom, full_size_left:right]
                binary_source_patch, patch_accuracy = optimise_threshold(
                    source=source_patch,
                    target=target_patch,
                    mask=mask_patch,
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
    input_bands: NDArray[Any], mosaic_device: Union[str, torch.device]
) -> torch.Tensor:
    input_bands_tensor = torch.from_numpy(input_bands.astype(np.float16)).to(
        mosaic_device
    )
    ndwi = (input_bands_tensor[1] - input_bands_tensor[3]) / (
        input_bands_tensor[1] + input_bands_tensor[3]
    )

    return ndwi


def make_composite_output(
    input_dict: dict[str, Optional[torch.Tensor]],
) -> tuple[NDArray[Any], list[str]]:
    output_layers = []
    layer_names = []
    # Get the shape of the first non-None layer
    shape = None
    for value in input_dict.values():
        if value is not None:
            shape = value.shape
            break
    if shape is None:
        raise ValueError("make_composite_output requires at least one non-None layer")
    for key, value in input_dict.items():
        #  if value is None, use a zero tensor to avoid missing layers
        if value is None:
            logging.info(f"Layer {key} is None, setting to zero tensor")
            value = torch.zeros(shape, dtype=torch.float32)
        output_layers.append(value.float().numpy(force=True).astype(np.float32))
        layer_names.append(key)
    return np.stack(output_layers), layer_names


def _collect_target(
    thread: Thread,
    result_queue: Queue[Any],
    name: str,
) -> Optional[torch.Tensor]:
    """Join a target-building thread and take its result.

    A failed build arrives as the exception itself, since raising inside the
    thread would be lost; re-raise it here so the scene is skipped rather than
    exported without its vector targets. A None means there was nothing to
    build, which is not a failure. An empty queue means ``build_targets`` raised
    before it could report - a bad argument, or an error outside its own try.
    Reading the queue blind would block forever on that, so surface it too.
    """
    if thread.is_alive():
        logging.info(f"Waiting for {name} targets to finish")
    thread.join()

    if result_queue.empty():
        raise RuntimeError(
            f"The {name} target thread exited without a result, meaning it raised "
            f"before it could report. That exception went to stderr through the "
            f"threading excepthook rather than through logging."
        )
    result: Union[torch.Tensor, BaseException, None] = result_queue.get()
    if isinstance(result, BaseException):
        raise result
    return result


def _try_collect_target(
    thread: Thread,
    result_queue: Queue[Any],
    name: str,
) -> tuple[Optional[torch.Tensor], Optional[BaseException]]:
    """Collect a target thread, returning its failure rather than raising it.

    Lets the caller join every thread before deciding what to propagate.
    """
    try:
        return _collect_target(thread, result_queue, name), None
    except Exception as e:
        return None, e


def integrate_water_detection_methods(
    input_bands: NDArray[Any],
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
    vector_source: str = "overture",
    include_ocean: bool = True,
    aux_vector_sources: Optional[list[Path]] = None,
    aux_negative_vector_sources: Optional[list[Path]] = None,
    mosaic_device: Union[str, torch.device] = "cpu",
    no_data_value: int = 0,
    optimise_model: bool = True,
) -> tuple[NDArray[Any], list[str], Optional[NDArray[Any]]]:
    """Combine the NDWI, model predictions and vector targets.

    Returns the stacked output array, the per-band layer names, and an optional
    validity mask (1 = valid, 0 = no data). The mask is ``None`` in debug mode
    (where the no-data layer is kept as a regular band); otherwise it is written
    as a GDAL dataset mask on export.
    """
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
    vector_target_result_queue: Queue[Any] = Queue()
    # The handles are owned here, not by build_targets, which is also called
    # directly with a dataset its caller goes on to reuse. They are closed in
    # the finally below, after both threads are joined - closing a dataset a
    # thread is still reading from would take that thread down with it.
    vector_raster_src = rio.open(input_path)
    negative_raster_src: Optional[rio.DatasetReader] = None
    vector_target_thread = Thread(
        target=build_targets,
        kwargs={
            "raster_src": vector_raster_src,
            "osm_water": use_osm_water,
            "aux_vector_sources": aux_vector_sources,
            "device": mosaic_device,
            "cache_dir": cache_dir,
            "use_cache": use_cache,
            "vector_source": vector_source,
            "include_ocean": include_ocean,
            # For now positive targets use all_touched=True (same as negative);
            # set to False to only flip pixels a feature mostly covers.
            "all_touched": True,
            "queue": vector_target_result_queue,
        },
    )
    vector_target_thread.start()

    negative_target_thread: Optional[Thread] = None
    negative_target_result_queue: Optional[Queue[Any]] = None
    vector_targets: Optional[torch.Tensor] = None
    vector_negative_target: Optional[torch.Tensor] = None

    try:
        if use_osm_building_mask or use_osm_roads_mask:
            logging.info("Building negative targets in thread")
            negative_raster_src = rio.open(input_path)
            negative_target_result_queue = Queue()
            negative_target_thread = Thread(
                target=build_targets,
                kwargs={
                    "raster_src": negative_raster_src,
                    "osm_buildings": use_osm_building_mask,
                    "osm_roads": use_osm_roads_mask,
                    "aux_vector_sources": aux_negative_vector_sources,
                    "device": mosaic_device,
                    "cache_dir": cache_dir,
                    "use_cache": use_cache,
                    "vector_source": vector_source,
                    "all_touched": True,
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

        vector_targets, vector_error = _try_collect_target(
            vector_target_thread, vector_target_result_queue, "vector"
        )

        negative_error: Optional[BaseException] = None
        if (
            negative_target_thread is not None
            and negative_target_result_queue is not None
        ):
            vector_negative_target, negative_error = _try_collect_target(
                negative_target_thread, negative_target_result_queue, "negative vector"
            )

        # Both threads are joined before either failure propagates. Raising as
        # soon as the first one failed would leave the other fetching on into
        # the next scene, and a widespread outage would pile those up across a
        # batch.
        if vector_error is not None:
            raise vector_error
        if negative_error is not None:
            raise negative_error
    finally:
        # Anything raised above - model inference, or a target failure being
        # propagated - can leave a thread still running, so join before closing
        # the datasets those threads read from.
        vector_target_thread.join()
        if negative_target_thread is not None:
            negative_target_thread.join()
        vector_raster_src.close()
        if negative_raster_src is not None:
            negative_raster_src.close()

    if vector_targets is not None:
        model_target.append(vector_targets)
        ndwi_target.append(vector_targets)

    if vector_negative_target is not None:
        negative_target.append(vector_negative_target)

    if len(negative_target) > 0:
        negative_target_tensor: Optional[torch.Tensor] = (
            torch.stack(negative_target).sum(0) > 0
        )
    else:
        negative_target_tensor = None

    ndwi_target_tensor: Optional[torch.Tensor]
    if use_ndwi:
        logging.info("Optimising NDWI")
        if len(ndwi_target) > 0:
            ndwi_target_tensor = torch.stack(ndwi_target).sum(0)
        else:
            ndwi_target_tensor = torch.zeros_like(ndwi_conf_tensor, dtype=torch.bool)

        (
            NDWI_binary,
            NDWI_accuracy_tracker,
            NDWI_cumulative_detections,
            _,
            normalised_accuracy,
        ) = multi_scale_optimisation(
            source=ndwi_conf_tensor,
            target=ndwi_target_tensor,
            patch_sizes=patch_sizes,
            mask=negative_target_tensor,
        )
        logging.info("Multi-scale optimisation accuracy finished")
        combined_water.append(NDWI_binary)
        model_target.append(NDWI_binary)
        model_target.append(ndwi_conf_tensor > 0.5)

    else:
        NDWI_binary = None
        ndwi_target_tensor = None
        NDWI_accuracy_tracker = None
        NDWI_cumulative_detections = None
        normalised_accuracy = None

    if len(model_target) > 0:
        model_target_tensor = torch.stack(model_target).sum(0)
    else:
        model_target_tensor = torch.zeros_like(ndwi_conf_tensor, dtype=torch.bool)

    if model_conf_tensor is not None:
        if optimise_model:
            logging.info("Optimising model predictions")
            model_binary_cleaned, _ = optimise_by_threshold_and_overlap(
                source=model_conf_tensor,
                target=model_target_tensor,
                mask=negative_target_tensor,
                scene_thresholds=(0, 1),
            )

            combined_water.append(model_binary_cleaned)
        else:
            logging.info("Using raw model predictions")
            assert model_binary is not None
            combined_water.append(model_binary)
            model_binary_cleaned = None

    else:
        model_conf_tensor = None
        model_binary_cleaned = None

    combined_water_tensor = torch.stack(combined_water).sum(0) > 0

    if debug_output:
        logging.info("Exporting debug layers")
        final_output, layer_names = make_composite_output(
            {
                "Water predictions": combined_water_tensor,
                "NDWI binary": NDWI_binary,
                "NDWI target": ndwi_target_tensor,
                "NDWI raw": ndwi_conf_tensor,
                "NDWI cumulative detections": NDWI_cumulative_detections,
                "NDWI accuracy tracker": NDWI_accuracy_tracker,
                "NDWI normalised accuracy": normalised_accuracy,
                "Model binary cleaned": model_binary_cleaned,
                "Model binary": model_binary,
                "Model target": model_target_tensor,
                "Model confidence": model_conf_tensor,
                "Vector inputs": vector_targets,
                "Negative vector inputs": negative_target_tensor,
                "No data mask": no_data_mask,
            }
        )
        nodata_mask_np = None
    else:
        final_output = combined_water_tensor.numpy(force=True).astype(np.uint8)
        final_output = np.expand_dims(final_output, axis=0)
        layer_names = ["Water predictions"]
        # validity mask: 1 where data is valid, 0 where no data. Written as a
        # GDAL dataset mask on export so QGIS treats nodata as transparent
        # rather than as a second grey data band.
        nodata_mask_np = (~(no_data_mask.bool())).numpy(force=True).astype(np.uint8)

    return final_output, layer_names, nodata_mask_np
