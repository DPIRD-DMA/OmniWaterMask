<div align="center">

<img src="https://raw.githubusercontent.com/DPIRD-DMA/OmniWaterMask/main/assets/omniwatermask-title.svg" alt="OmniWaterMask" width="680">

[![image](https://img.shields.io/pypi/v/omniwatermask.svg)](https://pypi.python.org/pypi/omniwatermask)
[![image](https://static.pepy.tech/badge/omniwatermask)](https://pepy.tech/project/omniwatermask)
[![image](https://img.shields.io/conda/vn/conda-forge/omniwatermask.svg)](https://anaconda.org/conda-forge/omniwatermask)
[![Conda Recipe](https://img.shields.io/badge/recipe-omniwatermask-green.svg)](https://github.com/conda-forge/omniwatermask-feedstock)

</div>

OmniWaterMask is a Python library for high accuracy water segmentation in high to moderate resolution satellite imagery, supporting a wide range of resolutions, sensors, and processing levels.

[Check out the paper here](https://www.sciencedirect.com/science/article/pii/S0924271625002692)


## Features

-   Process imagery resolutions from 0.2 m to 50 m.
-   Any imagery processing level
-   Only requires Red, Green, Blue and NIR bands
-   Known to work well with Sentinel-2, Landsat 8, PlanetScope, Maxar and NAIP

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/github/DPIRD-DMA/OmniWaterMask/blob/main/examples/Sentinel%202%20example.ipynb 'Try OmniWaterMask In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab

## How it works
OmniWaterMask integrates a sensor agnostic deep learning segmentation model with NDWI and vector datasets to detect water bodies within remote sensing products.

## Installation

To use OmniWaterMask, you need to install the package. It is recommended to use an environment manager such as conda or uv to avoid conflicts with other packages.

### Install the package using pip

```bash
pip install omniwatermask
```

### Install the package using uv

```bash
uv add omniwatermask
```

### Create a new conda environment and install from conda-forge

```bash
conda create -n owm python=3.12
conda activate owm
conda install -c conda-forge omniwatermask
```

### Install the package from source

```bash
pip install git+https://github.com/DPIRD-DMA/OmniWaterMask.git
```


## Usage

To predict a water mask for a list of scenes simply pass a list of geotiff files to the make_water_mask function along with the band order for the Red, Green, Blue and NIR bands. Predictions are saved to disk along side the input as geotiffs, a list of prediction file paths is returned:

```python
from pathlib import Path
from omniwatermask import make_water_mask

scene_paths = [Path("path/to/scene1.tif"), Path("path/to/scene2.tif")]

# Predict water masks for scenes
water_mask_path = make_water_mask(
    scene_paths=scene_paths,  # you can pass a list of images
    band_order=[1, 2, 3, 4],  # band order of the input images, expects RGB+NIR
)
```
## Output
- Output classes are:
- 0 = Non-water
- 1 = water

## Usage tips

-   OWM requires an active internet connection to function properly, as it needs to download vector data.
-   Which Overture release to read is resolved once and reused for the rest of the process, from Overture's release catalogue when it is reachable and otherwise from the newest release in Overture's S3 bucket that carries every theme OWM reads. It is rediscovered if a fetch later fails against it, so a long-running process picks up a new release after Overture prunes the old one. Releases are not pinned: Overture retains roughly two releases (~60 days) and prunes the rest, so a hardcoded release stops resolving within a couple of months. This also means an old run cannot be reproduced by pinning a release — the local vector cache is what makes a target set reproducible.
-   Vector data comes from [Overture Maps](https://overturemaps.org) by default. If you would rather query OpenStreetMap live through the Overpass API, set `vector_source="osm"`. Overture serves static monthly GeoParquet releases from cloud storage, so it avoids the rate limits and timeouts Overpass returns on large or dense bounding boxes. The underlying data is largely the same — Overture's water and road layers are derived from OSM — though its building footprints add machine-learning-derived data beyond OSM. Note that Overture files a few landforms (`cape`, `blowhole`, `shoal`) under its water theme; OWM filters these out so they are not treated as water.
-   If a scene's vector data cannot be fetched, that scene is skipped rather than processed without it — a mask built without its vector targets looks plausible but is quietly worse. Overture fetches retry transient failures first (3 attempts, 2s then 4s apart); failures that will not improve on a retry, such as being unable to determine which Overture release to read, are raised immediately instead of consuming the backoff. A skipped scene is logged at ERROR, is left out of the returned list of output paths, and has no file written, so re-running the same call reprocesses it while the rest of the batch is untouched. Because the two sources are largely interchangeable, an outage in one is worth trying the other for, and the error messages say so.
-   Hardware acceleration is strongly recommended:
    -   NVIDIA GPU
    -   Apple Silicon Mac
    -   Other PyTorch-compatible accelerators
-   Consider enabling "bf16" inference_dtype on compatible hardware - this typically results in faster processing speeds.
-   If experiencing VRAM limitations even with batch_size=1, switching the 'mosaic_device' parameter to 'cpu' can help.
-   Improve accuracy by providing known water body locations as 'aux_vector_sources' - simply pass a list of file paths pointing to your water polygon datasets.
-   Reduce false positives by including vector data for common misidentification sources (buildings, roads) through the 'aux_negative_vector_sources' parameter.

-   When working with scenes containing no-data regions, explicitly set the 'no_data_value' parameter to ensure proper handling of these areas.

### Cloudy imagery

If you are working with cloudy imagery, either:

-   use a **temporal mosaic** that is already cloud and cloud-shadow free (e.g. via [s2mosaic](https://github.com/DPIRD-DMA/s2mosaic) for Sentinel-2), or
-   apply a **high quality cloud and cloud shadow mask** and set those pixels to `0` (the `no_data_value`) before running OWM.

This matters because OWM optimises its detection thresholds both **locally** (per region/patch) and **globally** (across the whole scene). Cloud and cloud-shadow pixels are out-of-distribution and can skew those optimisations, so bad data in one part of a scene can degrade the water prediction in other, otherwise-clean parts. Masking those pixels to no-data removes them from the optimisation entirely.

[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) is a good choice for the masking step. See the [cloudy Sentinel-2 example](https://github.com/DPIRD-DMA/OmniWaterMask/blob/main/examples/Sentinel%202%20cloudy%20example.ipynb) for an end-to-end mask-then-infer workflow.


## Parameters

-    `scene_paths`: List of paths or single path (supports both Path and string types) to the input satellite/aerial imagery

-    `band_order`: List of integers specifying the band order for input imagery (e.g., [1,2,3,4] if your input image is stored with band order red, green, blue then NIR data). This tells OWM which bands correspond to Red, Green, Blue, and Near-Infrared channels

-    `batch_size`: Number of patches processed simultaneously during inference. Default is 1, increase for better GPU utilization

-    `version`: Version identifier for the output files. Defaults to current OmniWaterMask version

-    `output_dir`: Optional path for output files. If not specified, outputs are saved alongside input files

-    `mosaic_device`: Device for mosaic operations ("cpu", "cuda" or "mps"). Defaults to system's default device

-    `inference_device`: Device for model inference ("cpu", "cuda" or "mps"). Defaults to system's default device

-    `aux_vector_sources`: List of paths to supplementary water body vector data to aid detection

-    `aux_negative_vector_sources`: List of paths to vector data marking areas commonly misidentified as water

-    `inference_dtype`: Data type for inference operations. Defaults to torch.float32

-    `no_data_value`: Value indicating no-data regions in the input imagery. Defaults to 0

-    `inference_patch_size`: Size of image patches for inference. Defaults to 1000 pixels

-    `inference_overlap_size`: Overlap between adjacent patches during inference. Defaults to 300 pixels

-    `overwrite`: Whether to overwrite existing output files. Defaults to True

-    `use_cache`: Whether to cache vector data processing results. Defaults to True

-    `use_osm_building`: Whether to use building data to reduce false positives. Defaults to True

-    `use_osm_roads`: Whether to use road data to reduce false positives. Defaults to True

-    `vector_source`: Where water, road and building vectors come from — `"overture"` (Overture Maps GeoParquet) or `"osm"` (OpenStreetMap via the Overpass API). Defaults to "overture"

-    `include_ocean`: Whether Overture ocean polygons count as positive water targets. These cover everything seaward of the OSM coastline, which the OSM tag set does not provide. Set to False if coastline/tide offsets cause false positives on your scenes. Only applies when `vector_source="overture"`. Defaults to True

-    `cache_dir`: Directory for storing cached vector data. Defaults to "OWM_cache" in current directory

-    `destination_model_dir`: Directory to save the model weights. Defaults to None

-    `model_download_source`: Source from which to download the model weights. Defaults to "hugging_face", can also be "google_drive".


## Examples

Example notebooks are available in the [examples/](https://github.com/DPIRD-DMA/OmniWaterMask/tree/main/examples) directory:

-   [NAIP example](https://github.com/DPIRD-DMA/OmniWaterMask/blob/main/examples/NAIP%20example.ipynb) — Water segmentation on NAIP aerial imagery from HuggingFace
-   [Sentinel-2 example](https://github.com/DPIRD-DMA/OmniWaterMask/blob/main/examples/Sentinel%202%20example.ipynb) — Water segmentation on a Sentinel-2 mosaic using [s2mosaic](https://github.com/DPIRD-DMA/s2mosaic)
-   [Cloudy Sentinel-2 example](https://github.com/DPIRD-DMA/OmniWaterMask/blob/main/examples/Sentinel%202%20cloudy%20example.ipynb) — Masking clouds with [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) before running OWM on a cloudy AWS scene

## Changelog

See [CHANGELOG.md](https://github.com/DPIRD-DMA/OmniWaterMask/blob/main/CHANGELOG.md) for a full list of changes across versions.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

### Development setup

Clone the repository and install the dependencies (including the dev group) with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras --dev
```

Optionally install the git hooks (ruff lint/format on commit, mypy + the fast tests on push):

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### Running the tests

Tests use `pytest`. The fast suite (unit tests + model-mocked pipeline tests) runs in a few seconds and is what CI runs by default:

```bash
uv run pytest                              # full fast suite
uv run pytest tests/test_orchestration.py  # one file
uv run pytest -k make_water_mask           # match by name
```

End-to-end tests that download the real model weights and run inference on real imagery are marked `e2e` and excluded by default (see `addopts` in `pyproject.toml`). To run them explicitly:

```bash
uv run pytest -m e2e                        # only the e2e/inference tests
uv run pytest -m ""                         # everything, including e2e
```

Lint, format and type-check:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy omniwatermask/
```

For maintainers: pushing a version tag (e.g. `git tag v0.4.4 && git push --tags`) builds the package and publishes it to PyPI via GitHub Actions trusted publishing — no tokens required.

## License

This project is licensed under the MIT License

## Acknowledgements

Special thanks to the [S1S2-Water dataset authors ](https://github.com/MWieland/s1s2_water) and [The FLAIR #1 dataset authors](https://ignf.github.io/FLAIR/) for providing the valuable training datasets.
