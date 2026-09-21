import pytest
import torch

from omniwatermask import inference_dtype as module
from omniwatermask.inference_dtype import (
    fastest_inference_dtype,
    resolve_inference_dtype,
)


@pytest.fixture(autouse=True)
def clear_selection_cache():
    """Each test measures afresh; the selection is cached for a whole process."""
    module._fastest_inference_dtype_cached.cache_clear()
    yield
    module._fastest_inference_dtype_cached.cache_clear()


def fake_screen(timings):
    """_run_once stand-in driven by a dtype->seconds map."""

    def timer(device, dtype, size):
        return timings.get(dtype)

    return timer


def fake_measure(timings):
    """_measure stand-in driven by a dtype->seconds map."""

    def timer(device, dtype):
        return timings.get(dtype)

    return timer


class TestResolveInferenceDtype:
    def test_passes_through_torch_dtype(self):
        assert resolve_inference_dtype(torch.float16, "cpu") is torch.float16

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("bf16", torch.bfloat16),
            ("float16", torch.float16),
            ("fp32", torch.float32),
            ("torch.bfloat16", torch.bfloat16),
        ],
    )
    def test_passes_through_strings(self, given, expected):
        assert resolve_inference_dtype(given, "cpu") is expected

    def test_rejects_unknown_string(self):
        with pytest.raises(ValueError):
            resolve_inference_dtype("float8", "cpu")

    def test_auto_is_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(
            module, "fastest_inference_dtype", lambda device: torch.bfloat16
        )
        assert resolve_inference_dtype("AUTO", "cpu") is torch.bfloat16

    def test_auto_returns_a_real_dtype_on_cpu(self):
        """The unmocked path must work on whatever CPU the tests run on."""
        assert resolve_inference_dtype("auto", "cpu") in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }


class TestFastestInferenceDtype:
    def test_prefers_bfloat16_when_tied_with_float16(self, monkeypatch):
        """Equal speed is the normal case, so the numerically safer one wins.

        float16 and bfloat16 run on the same reduced-precision hardware path, so
        a device that accelerates one generally accelerates the other by the
        same amount. bfloat16 keeps float32's exponent range, so it is preferred
        unless float16 is clearly faster.
        """
        timings = {torch.float32: 0.100, torch.bfloat16: 0.050, torch.float16: 0.049}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))
        monkeypatch.setattr(module, "_measure", fake_measure(timings))
        assert fastest_inference_dtype("cpu") is torch.bfloat16

    def test_float16_used_only_when_bfloat16_is_unavailable(self, monkeypatch):
        timings = {torch.float32: 0.100, torch.bfloat16: None, torch.float16: 0.020}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))
        monkeypatch.setattr(module, "_measure", fake_measure(timings))
        assert fastest_inference_dtype("cpu") is torch.float16

    def test_falls_back_to_float32_when_gain_is_marginal(self, monkeypatch):
        """A few percent is within noise and not worth the precision loss."""
        timings = {torch.float32: 0.100, torch.bfloat16: 0.098, torch.float16: 0.097}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))
        monkeypatch.setattr(module, "_measure", fake_measure(timings))
        assert fastest_inference_dtype("cpu") is torch.float32

    def test_emulated_dtypes_are_screened_out(self, monkeypatch):
        """The case this exists for: accepted by the device, but far slower."""
        timings = {torch.float32: 0.002, torch.bfloat16: 0.185, torch.float16: 0.237}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))

        def fail(*args, **kwargs):
            raise AssertionError("screened-out dtypes must not be measured again")

        monkeypatch.setattr(module, "_measure", fail)
        assert fastest_inference_dtype("cpu") is torch.float32

    def test_unsupported_dtype_is_skipped(self, monkeypatch):
        timings = {torch.float32: 0.100, torch.bfloat16: None, torch.float16: 0.020}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))
        monkeypatch.setattr(module, "_measure", fake_measure(timings))
        assert fastest_inference_dtype("cpu") is torch.float16

    def test_float32_failure_falls_back_to_float32(self, monkeypatch):
        monkeypatch.setattr(module, "_run_once", fake_screen({}))
        assert fastest_inference_dtype("cpu") is torch.float32

    def test_result_is_cached_per_device(self, monkeypatch):
        """A batch of scenes must not re-measure once per scene."""
        calls = []
        timings = {torch.float32: 0.100, torch.bfloat16: 0.010, torch.float16: 0.010}

        def counting_screen(device, dtype, size):
            calls.append(dtype)
            return timings.get(dtype)

        def counting_measure(device, dtype):
            calls.append(dtype)
            return timings.get(dtype)

        monkeypatch.setattr(module, "_run_once", counting_screen)
        monkeypatch.setattr(module, "_measure", counting_measure)
        assert fastest_inference_dtype("cpu") is torch.bfloat16
        after_first = len(calls)
        for _ in range(5):
            assert fastest_inference_dtype("cpu") is torch.bfloat16
        assert len(calls) == after_first

    def test_accepts_device_objects_and_strings_alike(self, monkeypatch):
        timings = {torch.float32: 0.100, torch.bfloat16: 0.010, torch.float16: 0.010}
        monkeypatch.setattr(module, "_run_once", fake_screen(timings))
        monkeypatch.setattr(module, "_measure", fake_measure(timings))
        assert fastest_inference_dtype(torch.device("cpu")) is torch.bfloat16
        assert fastest_inference_dtype("cpu") is torch.bfloat16


class TestMeasurementLength:
    def test_repeats_until_the_window_is_long_enough(self, monkeypatch):
        """A single conv is a few ms on a GPU, where jitter picks the winner."""
        seen = []

        def timed_run(device, data, weight, iterations):
            seen.append(iterations)
            return 0.0005 * iterations

        monkeypatch.setattr(module, "_timed_run", timed_run)
        elapsed = module._measure(torch.device("cpu"), torch.float32)
        assert elapsed == pytest.approx(0.0005)
        # one warm-up call, then enough repeats to fill _MEASURE_SECONDS
        assert seen[0] == 1
        assert seen[1] * 0.0005 >= module._MEASURE_SECONDS

    def test_iteration_count_is_capped(self, monkeypatch):
        monkeypatch.setattr(
            module,
            "_timed_run",
            lambda device, data, weight, iterations: 1e-9 * iterations,
        )
        module._measure(torch.device("cpu"), torch.float32)

    def test_measure_returns_none_when_dtype_unusable(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("dtype not supported here")

        monkeypatch.setattr(module, "_probe_tensors", boom)
        assert module._measure(torch.device("cpu"), torch.bfloat16) is None


class TestChoose:
    def test_ties_go_to_the_preferred_dtype(self):
        timings = {torch.bfloat16: 0.050, torch.float16: 0.049}
        assert module._choose(timings, baseline=0.100) is torch.bfloat16

    def test_preference_holds_even_when_float16_probes_faster(self):
        """The probe cannot rank two dtypes on the same hardware path.

        It put float16 well ahead of bfloat16 on this machine while the real
        model ranked them the other way round, so the probe only decides
        whether to drop precision, never which dtype to drop to.
        """
        timings = {torch.bfloat16: 0.080, torch.float16: 0.020}
        assert module._choose(timings, baseline=0.100) is torch.bfloat16

    def test_float16_used_when_bfloat16_will_not_run(self):
        assert module._choose({torch.float16: 0.020}, baseline=0.100) is torch.float16

    def test_marginal_gain_keeps_float32(self):
        timings = {torch.bfloat16: 0.098, torch.float16: 0.097}
        assert module._choose(timings, baseline=0.100) is torch.float32

    def test_no_timings_keeps_float32(self):
        assert module._choose({}, baseline=0.100) is torch.float32
