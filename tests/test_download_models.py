from pathlib import Path
from unittest.mock import patch

import pytest

from omniwatermask.download_models import (
    _model_index,
    download_file,
    download_file_from_hugging_face,
    get_latest_model_version,
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
        # Pre-create a fake model file large enough to skip re-download. Named
        # from the index so a new model version does not turn this into a test
        # that an absent file is not downloaded.
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        index = _model_index()
        latest = index[index["version"] == get_latest_model_version()]
        for model_name in latest["file_name"]:
            (model_dir / str(model_name)).write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

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
        get_models(force_download=True, model_dir=model_dir, source="hugging_face")
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
            "PM_model_1.5.38_s1s2_water_flair_convnextv2_base_PT.pth_weights.pth"
        )
        fake_model = model_dir / model_name
        fake_model.write_bytes(b"x" * 100)  # tiny file

        get_models(model_dir=model_dir, source="hugging_face")
        assert mock_download.called


class TestHuggingFaceDownloadShapes:
    """The Hub always serves safetensors; the entry's own suffix decides the rest.

    A v2+ entry names the safetensors itself, so the same published file backs
    both the Hub and the Google Drive copy and nothing is converted. A v1 entry
    names a ``.pth`` because that is what its Drive copy is, so the safetensors
    has to be rewritten as a torch state to keep one file name per entry.
    """

    @patch("omniwatermask.download_models.torch.save")
    @patch("omniwatermask.download_models.load_file")
    @patch("omniwatermask.download_models.hf_hub_download")
    def test_safetensors_entry_is_not_converted(
        self, mock_hf, mock_load, mock_save, tmp_path
    ):
        dest = tmp_path / "PM_model_2.3.4_smp_convnextv2_nano_PT_state.safetensors"
        mock_hf.return_value = str(dest)

        download_file_from_hugging_face(dest)

        # Asked the Hub for the entry's own name, and left the file alone.
        assert mock_hf.call_args.kwargs["filename"] == (
            "PM_model_2.3.4_smp_convnextv2_nano_PT_state.safetensors"
        )
        mock_load.assert_not_called()
        mock_save.assert_not_called()

    @patch("omniwatermask.download_models.torch.save")
    @patch("omniwatermask.download_models.load_file")
    @patch("omniwatermask.download_models.hf_hub_download")
    def test_pth_entry_is_converted(self, mock_hf, mock_load, mock_save, tmp_path):
        dest = tmp_path / "PM_model_1.5.38_convnextv2_base_PT.pth_weights.pth"
        mock_hf.return_value = str(tmp_path / "downloaded.safetensors")
        mock_load.return_value = {"weight": "tensor"}

        download_file_from_hugging_face(dest)

        assert mock_hf.call_args.kwargs["filename"] == (
            "PM_model_1.5.38_convnextv2_base_PT.pth_weights.safetensors"
        )
        mock_save.assert_called_once_with({"weight": "tensor"}, dest)

    @patch("omniwatermask.download_models.torch.save")
    @patch("omniwatermask.download_models.load_file")
    @patch("omniwatermask.download_models.hf_hub_download")
    def test_download_lands_at_the_destination(
        self, mock_hf, mock_load, mock_save, tmp_path
    ):
        """Without local_dir the file lands in a cache subdirectory instead."""
        dest = tmp_path / "model_PT_state.safetensors"
        mock_hf.return_value = str(dest)

        download_file_from_hugging_face(dest)

        assert mock_hf.call_args.kwargs["local_dir"] == tmp_path


class TestModelVersionSelection:
    """Which model an unpinned caller gets, and how an old one is asked for.

    The index carries more than one generation, and they are not
    interchangeable: version 1 is a fastai model and needs the legacy extra,
    version 2 onward are smp. A caller that pins nothing must get the newest,
    or a released model would sit unused until someone passed a number.
    """

    def test_latest_is_the_highest_version_in_the_index(self):
        index = _model_index()
        assert get_latest_model_version() == index["version"].max()

    @patch("omniwatermask.download_models.download_file")
    def test_unpinned_resolves_to_the_latest(self, mock_download, tmp_path):
        index = _model_index()
        expected = set(
            index[index["version"] == get_latest_model_version()]["file_name"]
        )

        result = get_models(model_dir=tmp_path, source="hugging_face")

        assert {Path(r["Path"]).name for r in result} == expected

    @patch("omniwatermask.download_models.download_file")
    def test_an_older_version_can_be_pinned(self, mock_download, tmp_path):
        index = _model_index()
        oldest = float(index["version"].min())
        expected = set(index[index["version"] == oldest]["file_name"])

        result = get_models(
            model_dir=tmp_path, source="hugging_face", model_version=oldest
        )

        assert {Path(r["Path"]).name for r in result} == expected
        assert result != []

    @patch("omniwatermask.download_models.download_file")
    def test_library_travels_with_the_entry(self, mock_download, tmp_path):
        """model_library decides which architecture the weights are built into."""
        index = _model_index()
        for _, row in index.iterrows():
            result = get_models(
                model_dir=tmp_path,
                source="hugging_face",
                model_version=float(row["version"]),
            )
            entry = next(r for r in result if Path(r["Path"]).name == row["file_name"])
            assert entry["model_library"] == row["model_library"]
            assert entry["timm_model_name"] == row["timm_model_name"]


class TestPublishedIndex:
    """The index is what a released install reads; a bad row fails at download."""

    def test_smp_entries_use_a_timm_universal_encoder(self):
        """smp's own encoder list has no convnextv2; tu- routes it through timm."""
        index = _model_index()
        for _, row in index[index["model_library"] == "smp"].iterrows():
            assert str(row["timm_model_name"]).startswith("tu-"), row["file_name"]

    def test_every_entry_names_a_loadable_suffix(self):
        index = _model_index()
        for name in index["file_name"]:
            assert Path(str(name)).suffix in {".pth", ".safetensors"}, name

    def test_versions_are_unique_per_generation(self):
        index = _model_index()
        counts = index.groupby("version")["model_library"].nunique()
        assert (counts == 1).all(), "a version mixes model libraries"
