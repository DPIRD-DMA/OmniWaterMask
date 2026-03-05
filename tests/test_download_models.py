from pathlib import Path
from unittest.mock import patch

import pytest

from omniwatermask.download_models import (
    download_file,
    get_model_data_dir,
    get_models,
)


class TestGetModelDataDir:
    def test_returns_path(self):
        result = get_model_data_dir()
        assert isinstance(result, Path)

    def test_directory_exists(self):
        result = get_model_data_dir()
        assert result.exists()


class TestDownloadFile:
    @patch("omniwatermask.download_models.download_file_from_google_drive")
    def test_google_drive_source(self, mock_download, tmp_path):
        dest = tmp_path / "model.pth"
        download_file("some_id", dest, "google_drive")
        mock_download.assert_called_once_with("some_id", dest)

    @patch("omniwatermask.download_models.download_file_from_hugging_face")
    def test_hugging_face_source(self, mock_download, tmp_path):
        dest = tmp_path / "model.pth"
        download_file("some_id", dest, "hugging_face")
        mock_download.assert_called_once_with(dest)

    def test_invalid_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid source"):
            download_file("id", tmp_path / "model.pth", "invalid_source")


class TestGetModels:
    @patch("omniwatermask.download_models.download_file")
    def test_returns_model_paths(self, mock_download, tmp_path):
        # Pre-create a fake model file large enough to skip re-download
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_name = (
            "PM_model_1.5.38_s1s2_water_flair_"
            "convnextv2_base_PT.pth_weights.pth"
        )
        fake_model = model_dir / model_name
        fake_model.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        result = get_models(model_dir=model_dir, source="hugging_face")
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Path" in result[0]
        assert "timm_model_name" in result[0]
        assert "model_library" in result[0]
        # download should not be called since file exists and is > 1MB
        mock_download.assert_not_called()

    @patch("omniwatermask.download_models.download_file")
    def test_force_download(self, mock_download, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        get_models(
            force_download=True, model_dir=model_dir, source="hugging_face"
        )
        assert mock_download.called

    def test_invalid_version_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            get_models(model_dir=tmp_path, model_version=999.0)

    @patch("omniwatermask.download_models.download_file")
    def test_downloads_small_file(self, mock_download, tmp_path):
        """Files under 1MB should be re-downloaded."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_name = (
            "PM_model_1.5.38_s1s2_water_flair_"
            "convnextv2_base_PT.pth_weights.pth"
        )
        fake_model = model_dir / model_name
        fake_model.write_bytes(b"x" * 100)  # tiny file

        get_models(model_dir=model_dir, source="hugging_face")
        assert mock_download.called
