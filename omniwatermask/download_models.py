from importlib import resources
from pathlib import Path
from typing import Any, Optional, Union

import gdown
import pandas as pd
import platformdirs
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

try:
    from ._version import __version__ as omniwatermask_version
except ImportError:
    omniwatermask_version = "0.0.0+unknown"

gdown_download: Any = getattr(gdown, "download")  # noqa: B009


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive using gdown.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The local path where the file should be saved.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown_download(url, str(destination), quiet=False)


def download_file_from_hugging_face(destination: Path) -> None:
    """
    Downloads a file from Hugging Face using hf_hub_download.

    Weights are published on the Hub as safetensors whatever the model
    generation, so ``local_dir`` puts the download at ``destination``
    directly and a safetensors-named entry needs nothing further. A v1
    entry names a ``.pth``, which is what the Google Drive copy of that
    generation is, so it is converted to keep one file name per entry
    across both sources.

    Args:
        destination (Path): The local path where the file should
            be saved.
    """
    file_name = destination.stem
    safetensor_path = hf_hub_download(
        repo_id="NickWright/OmniWaterMask",
        filename=f"{file_name}.safetensors",
        force_download=True,
        cache_dir=destination.parent,
        local_dir=destination.parent,
    )
    if destination.suffix == ".pth":
        model_state = load_file(safetensor_path)
        torch.save(model_state, destination)


def download_file(file_id: str, destination: Path, source: str) -> None:
    if source == "google_drive":
        download_file_from_google_drive(file_id, destination)
    elif source == "hugging_face":
        download_file_from_hugging_face(destination)
    else:
        raise ValueError(
            "Invalid source. Supported sources are 'google_drive' and 'hugging_face'."
        )


def _release_version(version: str) -> str:
    """Return only the public release portion of a version string.

    Tag-based versioning produces dev/dirty suffixes between releases
    (e.g. "0.5.1.dev3+g1a2b3c4"). Keying the model cache on the full
    string would create a new empty directory for every commit and force
    a re-download. Stripping the ".dev*"/"+local" suffix keeps the cache
    stable between releases while still refreshing on real version bumps.
    """
    return version.split("+")[0].split(".dev")[0]


def _model_index() -> "pd.DataFrame":
    """Read the packaged model index, with versions as floats."""
    with (resources.files("omniwatermask") / "model_download_links.csv").open() as f:
        model_df = pd.read_csv(f)
    model_df["version"] = model_df["version"].astype(float)
    return model_df


def get_latest_model_version() -> float:
    """Highest model version in the packaged index."""
    return float(_model_index()["version"].max())


def get_model_data_dir() -> Path:
    """Get the user data directory for model files"""
    data_dir = Path(
        platformdirs.user_data_dir(
            "omniwatermask",
            version=_release_version(omniwatermask_version),
            ensure_exists=True,
        )
    )
    return data_dir


def get_models(
    force_download: bool = False,
    model_dir: Union[str, Path, None] = None,
    source: str = "hugging_face",
    model_version: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Downloads the model weights and saves them locally.

    Args:
        force_download (bool): Whether to force download the model
            weights even if they already exist locally.
        model_dir (Union[str, Path, None]): The directory where the
            model weights should be saved.
        source (str): The source from which to download. Currently
            only "google_drive" or "hugging_face" are supported.
        model_version (Optional[float]): Which model version to fetch.
            Defaults to the highest version in the packaged index.
            Versions below 2 are fastai models and need the "legacy"
            extra installed.
    """

    model_df = _model_index()

    if model_version is None:
        model_version = get_latest_model_version()

    available_versions = model_df["version"].unique()
    if model_version not in available_versions:
        raise ValueError(
            f"""Model version {model_version} not found. 
            Available versions: {available_versions}"""
        )
    # filter models by version
    model_df = model_df[model_df["version"] == model_version]

    model_paths = []

    if model_dir is not None:
        model_dir = Path(model_dir)
    else:
        model_dir = get_model_data_dir()

    for _, row in model_df.iterrows():
        file_id = str(row["google_drive_id"])

        model_dir.mkdir(exist_ok=True)
        destination = model_dir / str(row["file_name"])
        timm_model_name = row["timm_model_name"]
        model_library = row["model_library"]

        if not destination.exists() or force_download:
            download_file(file_id=file_id, destination=destination, source=source)

        elif destination.stat().st_size <= 1024 * 1024:
            download_file(file_id=file_id, destination=destination, source=source)

        model_paths.append(
            {
                "Path": destination,
                "timm_model_name": timm_model_name,
                "model_library": model_library,
            }
        )
    return model_paths
