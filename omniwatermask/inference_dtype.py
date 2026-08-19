"""Pick an inference dtype by measuring the device rather than assuming.

Whether a reduced-precision dtype is worth using is not something the device
reports honestly. A device can accept ``bfloat16`` tensors and run every kernel
on them while emulating the arithmetic in software, which is slower than the
``float32`` it was meant to beat - so a capability flag is not evidence of a
speedup. The only reliable signal is a timed comparison on the device that will
actually run the model, which is what this module does, once per process.
"""

import logging
import math
import time
from functools import lru_cache
from typing import Union, cast

import torch
from omnicloudmask.model_utils import get_torch_dtype

# Ordered by preference. bfloat16 leads because it keeps float32's exponent
# range, so it degrades gracefully where float16 would overflow to inf.
_CANDIDATE_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)

# A reduced-precision dtype has to beat float32 by this margin before it is
# used at all. Below it the measurement is within noise, and float32 is the
# safer default - it is what the models were trained and validated in.
_REQUIRED_SPEEDUP = 1.10

# Each measurement repeats the convolution until it has run for at least this
# long, then divides. A single convolution takes a few milliseconds on a GPU,
# where scheduling jitter is a large fraction of the total and the winner
# changes from run to run - measured across five processes, an earlier version
# of this returned float16, bfloat16, float32, bfloat16 and float32. Repeating
# is far cheaper than enlarging the tensor, which grows memory quadratically.
_MEASURE_SECONDS = 0.050
_MEASURE_MAX_ITERATIONS = 500

# Small enough to stay cheap on a CPU, large enough that a GPU is doing real
# work per call rather than only launching kernels.
_PROBE_SIZE = 512
_PROBE_CHANNELS = 32

# Screening runs at this size, one call per dtype, purely to spot a dtype the
# device emulates before it is measured properly.
_SCREEN_SIZE = 128


def resolve_inference_dtype(
    inference_dtype: Union[torch.dtype, str],
    inference_device: Union[torch.device, str],
) -> torch.dtype:
    """Resolve ``inference_dtype``, measuring the device when it is ``"auto"``.

    Anything else is passed through to omnicloudmask's ``get_torch_dtype``
    unchanged, so an explicit choice is never second-guessed.
    """
    if isinstance(inference_dtype, str) and inference_dtype.lower() == "auto":
        return fastest_inference_dtype(inference_device)
    return cast(torch.dtype, get_torch_dtype(inference_dtype))


def fastest_inference_dtype(
    inference_device: Union[torch.device, str],
) -> torch.dtype:
    """Return the fastest usable dtype for ``inference_device``.

    Falls back to ``float32`` whenever a candidate cannot run, or fails to beat
    float32 by ``_REQUIRED_SPEEDUP``.
    """
    device = torch.device(inference_device)
    # Cached on the device's identity rather than the object, so a batch of
    # scenes measures once rather than once per scene.
    return _fastest_inference_dtype_cached(device.type, device.index)


@lru_cache(maxsize=None)
def _fastest_inference_dtype_cached(
    device_type: str, device_index: Union[int, None]
) -> torch.dtype:
    if device_index is None:
        device = torch.device(device_type)
    else:
        device = torch.device(device_type, device_index)

    contenders = _screen(device)
    if not contenders:
        logging.info(
            "Auto inference dtype: float32 on %s (nothing else ran faster)", device
        )
        return torch.float32

    baseline = _measure(device, torch.float32)
    if baseline is None:
        # If float32 itself will not run there is nothing to compare against,
        # and nothing better to suggest.
        logging.debug("Could not measure float32 on %s, using it", device)
        return torch.float32

    timings: dict[torch.dtype, float] = {}
    for candidate in contenders:
        elapsed = _measure(device, candidate)
        if elapsed is None:
            logging.debug("%s is unusable on %s", candidate, device)
            continue
        timings[candidate] = elapsed
        logging.debug(
            "%s on %s: %.2f ms (%.2fx vs float32)",
            candidate,
            device,
            elapsed * 1000,
            baseline / elapsed,
        )

    chosen = _choose(timings, baseline)
    if chosen is torch.float32:
        logging.info(
            "Auto inference dtype: float32 on %s (nothing faster by %.0f%%)",
            device,
            (_REQUIRED_SPEEDUP - 1) * 100,
        )
    else:
        logging.info(
            "Auto inference dtype: %s on %s (%.2fx faster than float32)",
            chosen,
            device,
            baseline / timings[chosen],
        )
    return chosen


def _choose(timings: dict[torch.dtype, float], baseline: float) -> torch.dtype:
    """Decide whether to drop precision, and if so which dtype to drop to.

    These are deliberately two different kinds of question. Whether to drop at
    all is decided by measurement, comparing the fastest candidate against
    float32 - that difference is large and consistent, an order of magnitude on
    a device that emulates.

    Which dtype to drop to is decided by preference, not measurement. A single
    convolution does not predict how a whole model will rank two dtypes that
    share the same hardware path: probing this machine put float16 ahead of
    bfloat16 (0.6 ms against 1.0-2.0 ms, and bfloat16's figure moved by 2x
    between runs), while running the actual model put bfloat16 ahead (21.1 s
    against 23.3 s). A probe that cannot reproduce the answer it is meant to
    predict should not be casting the deciding vote, so bfloat16 wins on its
    numerics - it keeps float32's exponent range, where float16 overflows to
    inf - and float16 is used only where bfloat16 will not run.
    """
    if not timings:
        return torch.float32

    if baseline / min(timings.values()) < _REQUIRED_SPEEDUP:
        return torch.float32

    for candidate in _CANDIDATE_DTYPES:
        if candidate in timings:
            return candidate
    return torch.float32


def _screen(device: torch.device) -> list[torch.dtype]:
    """Drop candidates that are obviously not worth measuring properly.

    A device that emulates a dtype in software can be hundreds of times slower
    in it, and measuring such a candidate at the full probe size wastes that
    entire factor - on a CPU it turned a sub-second selection into 212 s. So
    each candidate gets one small call first, and anything already slower than
    float32 there is dropped: it cannot go on to beat it by
    ``_REQUIRED_SPEEDUP``.
    """
    reference = _run_once(device, torch.float32, _SCREEN_SIZE)
    if reference is None:
        return []

    contenders = []
    for candidate in _CANDIDATE_DTYPES:
        elapsed = _run_once(device, candidate, _SCREEN_SIZE)
        if elapsed is None:
            logging.debug("%s is unusable on %s", candidate, device)
        elif elapsed > reference:
            logging.debug(
                "%s screened out on %s: %.1f ms against float32's %.1f ms",
                candidate,
                device,
                elapsed * 1000,
                reference * 1000,
            )
        else:
            contenders.append(candidate)
    return contenders


def _measure(device: torch.device, dtype: torch.dtype) -> Union[float, None]:
    """Time one convolution in ``dtype``, or return None if it will not run.

    A convolution rather than a matmul because that is where the segmentation
    models spend their time, and a device can accelerate one without the other.
    """
    try:
        data, weight = _probe_tensors(device, dtype, _PROBE_SIZE)
        with torch.inference_mode():
            # The first call pays for kernel compilation and allocator setup,
            # which is not what is being compared.
            single = _timed_run(device, data, weight, iterations=1)
            iterations = min(
                _MEASURE_MAX_ITERATIONS,
                max(1, math.ceil(_MEASURE_SECONDS / max(single, 1e-9))),
            )
            return _timed_run(device, data, weight, iterations) / iterations
    except Exception:
        # An unsupported dtype surfaces as anything from TypeError to a device
        # specific RuntimeError, and an unusable candidate is not an error worth
        # propagating - it just means float32 stands.
        return None


def _run_once(
    device: torch.device, dtype: torch.dtype, size: int
) -> Union[float, None]:
    """One call, for screening, where precision does not matter."""
    try:
        data, weight = _probe_tensors(device, dtype, size)
        with torch.inference_mode():
            return _timed_run(device, data, weight, iterations=1)
    except Exception:
        return None


def _probe_tensors(
    device: torch.device, dtype: torch.dtype, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.zeros((1, _PROBE_CHANNELS, size, size), device=device, dtype=dtype)
    weight = torch.zeros(
        (_PROBE_CHANNELS, _PROBE_CHANNELS, 3, 3), device=device, dtype=dtype
    )
    return data, weight


def _timed_run(
    device: torch.device, data: torch.Tensor, weight: torch.Tensor, iterations: int
) -> float:
    """Run the convolution ``iterations`` times and return the total seconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        torch.nn.functional.conv2d(data, weight, padding=1)
    _synchronize(device)
    return time.perf_counter() - start


def _synchronize(device: torch.device) -> None:
    """Wait for queued work, so the timing covers it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
