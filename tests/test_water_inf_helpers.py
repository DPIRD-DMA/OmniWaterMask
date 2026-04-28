import numpy as np
import torch

from omniwatermask.water_inf_helpers import (
    get_masked_iou,
    get_NDWI,
    make_composite_output,
    optimise_threshold,
    get_intersection_ratio,
    optimise_by_threshold_and_overlap,
    multi_scale_optimisation,
)


class TestGetMaskedIou:
    def test_perfect_overlap(self):
        source = torch.ones(10, 10, dtype=torch.bool)
        target = torch.ones(10, 10, dtype=torch.bool)
        iou = get_masked_iou(source, target, weighted=False)
        assert iou == 1.0

    def test_no_overlap(self):
        source = torch.ones(10, 10, dtype=torch.bool)
        target = torch.zeros(10, 10, dtype=torch.bool)
        iou = get_masked_iou(source, target, weighted=False)
        assert iou == 0.0

    def test_partial_overlap(self):
        source = torch.zeros(10, 10, dtype=torch.bool)
        target = torch.zeros(10, 10, dtype=torch.bool)
        source[:5, :] = True
        target[:5, :] = True
        iou = get_masked_iou(source, target, weighted=False)
        assert iou == 1.0  # perfect overlap on the 5-row region

    def test_with_mask(self):
        source = torch.ones(10, 10, dtype=torch.bool)
        target = torch.ones(10, 10, dtype=torch.bool)
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[:5, :] = True  # mask out top half
        iou = get_masked_iou(source, target, mask=mask, weighted=False)
        assert iou == 1.0  # unmasked region still matches

    def test_empty_tensors(self):
        source = torch.zeros(10, 10, dtype=torch.bool)
        target = torch.zeros(10, 10, dtype=torch.bool)
        iou = get_masked_iou(source, target, weighted=False)
        assert iou == 0.0  # union is 0

    def test_weighted_iou(self):
        source = torch.ones(10, 10, dtype=torch.bool)
        target = torch.ones(10, 10, dtype=torch.float32) * 2
        iou = get_masked_iou(source, target, weighted=True)
        assert iou > 0


class TestOptimiseThreshold:
    def test_finds_optimal_threshold(self):
        # Source with known threshold: values > 0.0 should match target
        source = torch.linspace(-1, 1, 100).reshape(10, 10)
        target = (source > 0.0).float()
        result, accuracy = optimise_threshold(source, target, mask=None)
        assert accuracy > 0.8
        assert result.shape == (10, 10)

    def test_with_mask(self):
        source = torch.linspace(-1, 1, 100).reshape(10, 10)
        target = (source > 0.0).float()
        mask = torch.zeros(10, 10, dtype=torch.bool)
        result, accuracy = optimise_threshold(source, target, mask=mask)
        assert accuracy > 0.8


class TestGetIntersectionRatio:
    def test_full_intersection(self):
        source = torch.zeros(20, 20, dtype=torch.bool)
        source[5:15, 5:15] = True
        target = torch.zeros(20, 20, dtype=torch.bool)
        target[5:15, 5:15] = True
        ratios = get_intersection_ratio(source, target)
        assert ratios.shape == (20, 20)
        # The cluster fully intersects with target → ratio ~1.0
        assert ratios[10, 10] > 0.9

    def test_no_intersection(self):
        source = torch.zeros(20, 20, dtype=torch.bool)
        source[0:5, 0:5] = True
        target = torch.zeros(20, 20, dtype=torch.bool)
        target[15:20, 15:20] = True
        ratios = get_intersection_ratio(source, target)
        # The cluster does not intersect with target → ratio ~0
        assert ratios[2, 2] < 0.1


class TestOptimiseByThresholdAndOverlap:
    def test_returns_two_tensors(self):
        source = torch.linspace(-1, 1, 400).reshape(20, 20)
        target = (source > 0).float()
        result, thresholded = optimise_by_threshold_and_overlap(
            source, target, mask=None
        )
        assert result.shape == (20, 20)
        assert thresholded.shape == (20, 20)


class TestMultiScaleOptimisation:
    def test_returns_five_tensors(self):
        source = torch.linspace(-1, 1, 2500).reshape(50, 50)
        target = (source > 0).float()
        results = multi_scale_optimisation(
            source=source,
            target=target,
            patch_sizes=[20],
            mask=None,
        )
        assert len(results) == 5
        for r in results:
            assert r.shape == (50, 50)


class TestGetNDWI:
    def test_ndwi_calculation(self):
        # bands: [Blue, Green, Red, NIR]
        bands = np.array(
            [
                [[100, 200]],  # Blue
                [[400, 600]],  # Green
                [[300, 500]],  # Red
                [[200, 800]],  # NIR
            ],
            dtype=np.float32,
        )
        result = get_NDWI(bands, mosaic_device="cpu")
        # NDWI = (Green - NIR) / (Green + NIR)
        expected_0 = (400 - 200) / (400 + 200)  # 0.333...
        expected_1 = (600 - 800) / (600 + 800)  # -0.142...
        assert abs(result[0, 0].item() - expected_0) < 0.01
        assert abs(result[0, 1].item() - expected_1) < 0.01

    def test_water_has_positive_ndwi(self):
        """Water pixels (high green, low NIR) should have positive NDWI."""
        bands = np.zeros((4, 10, 10), dtype=np.float32)
        bands[1] = 500  # Green
        bands[3] = 100  # NIR (low for water)
        result = get_NDWI(bands, "cpu")
        assert (result > 0).all()


class TestMakeCompositeOutput:
    def test_stacks_layers(self):
        layers = {
            "layer1": torch.ones(10, 10),
            "layer2": torch.zeros(10, 10),
        }
        output, names = make_composite_output(layers)
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0].shape == (10, 10)
        assert names == ["layer1", "layer2"]

    def test_handles_none_values(self):
        layers = {
            "present": torch.ones(10, 10),
            "missing": None,
        }
        output, names = make_composite_output(layers)
        assert isinstance(output, list)
        assert len(output) == 2
        assert np.all(output[1] == 0)

    def test_output_dtype_is_float32(self):
        layers = {"a": torch.ones(5, 5, dtype=torch.int32)}
        output, _ = make_composite_output(layers)
        assert output[0].dtype == np.float32
