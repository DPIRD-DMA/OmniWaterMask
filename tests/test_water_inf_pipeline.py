from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from omniwatermask.water_inf_pipeline import collect_models, make_water_mask_debug


class TestCollectModels:
    @patch("omniwatermask.water_inf_pipeline.load_model")
    def test_loads_single_custom_model(self, mock_load):
        mock_load.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path="/fake/model.pth",
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 1
        mock_load.assert_called_once()

    @patch("omniwatermask.water_inf_pipeline.load_model")
    def test_loads_multiple_custom_models(self, mock_load):
        mock_load.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path=["/fake/m1.pth", "/fake/m2.pth"],
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 2
        assert mock_load.call_count == 2

    @patch("omniwatermask.water_inf_pipeline.load_model_from_weights")
    @patch("omniwatermask.water_inf_pipeline.get_models")
    def test_loads_default_models(self, mock_get_models, mock_load_weights):
        mock_get_models.return_value = [
            {
                "Path": Path("/fake/model.pth"),
                "timm_model_name": "convnextv2_base",
                "model_library": "fastai",
            }
        ]
        mock_load_weights.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path="",
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 1
        mock_load_weights.assert_called_once()


class TestMakeWaterMaskDebug:
    def test_raises_without_model_or_ndwi_and_no_vectors(self, sample_geotiff):
        with pytest.raises(ValueError, match="must enable use_model"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=False,
                use_ndwi=True,
            )

    def test_raises_without_ndwi_and_no_vectors(self, sample_geotiff):
        with pytest.raises(ValueError, match="must enable use_ndwi"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=True,
                use_ndwi=False,
            )

    def test_accepts_string_scene_path(self, sample_geotiff):
        """Verify string paths are accepted (validation only, not full run)."""
        with pytest.raises(ValueError, match="must enable use_model"):
            make_water_mask_debug(
                scene_paths=str(sample_geotiff),
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=False,
                use_ndwi=True,
            )

    def test_rejects_invalid_scene_paths_type(self):
        with pytest.raises(ValueError, match="scene_paths must be"):
            make_water_mask_debug(
                scene_paths=123,
                band_order=[1, 2, 3, 4],
            )
