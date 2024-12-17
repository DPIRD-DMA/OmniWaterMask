# OmniWaterMask

OmniWaterMask is a Python library for high accuracy water segmentation in high to moderate resolution satellite imagery, supporting a wide range of resolutions, sensors, and processing levels.

## Features

-   Process imagery resolutions from 0.2 m to 50 m.
-   Any imagery processing level
-   Only requires Red, Green, Blue and NIR bands
-   Known to work well with Sentinel-2, Landsat 8, PlanetScope, Maxar and NAIP

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/drive/15qgTgk4XiWcSzllNt4CtFU7pDSNrKLHk?usp=sharing 'Try OmniWaterMask In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab

## How it works
OmniWaterMask integrates a sensor agnostic deep learning segmentation model with NDWI and vector datasets to detect water bodies within remote sensing products.

Paper coming soon...

## Installation

To install the package, use one of the following commands:

Make sure you are running Python 3.9 or above and  

```bash
conda create -n owm python=3.12
conda activate owm

pip install omniwatermask
```

```bash
conda create -n owm python=3.12
conda activate owm

pip install git+https://github.com/DPIRD-DMA/OmniWaterMask.git
```

## Usage

To predict a water mask for a list of scenes simply pass a list of geotiff files to the make_water_mask function along with the band order for the Red, Green, Blue and NIR bands. Predictions are saved to disk along side the input as geotiffs, a list of predictions file paths is returned:

```python
from pathlib import Path
from omniwatermask import make_water_mask

scene_paths = [Path("path/to/scene1.tif"), Path("path/to/scene2.tif")]

# Predict water masks for scenes
water_mask_path = make_water_mask(
    scene_paths=[scene_paths],  # you can pass a list of images
    band_order=[1, 2, 3, 4],  # band order of the input images, expects RGB+NIR
)
```
## Output
- Output classes are:
- 0 = Non-water
- 1 = water

## Usage tips

-   OWM requires an active internet connection to function properly, as it needs to download OpenStreetMap (OSM) data.
-   Hardware acceleration is strongly recommended:
    -   NVIDIA GPU
    -   Apple Silicon Mac
    -   Other PyTorch-compatible accelerators
-   Consider enabling "bf16" inference_dtype on compatible hardware - this typically results in faster processing speeds.
-   If experiencing VRAM limitations even with batch_size=1, switching the 'mosaic_device' parameter to 'cpu' can help.
-   Improve accuracy by providing known water body locations as 'aux_vector_sources' - simply pass a list of file paths pointing to your water polygon datasets.
-   Reduce false positives by including vector data for common misidentification sources (buildings, roads) through the 'aux_negative_vector_sources' parameter.

-   When working with scenes containing no-data regions, explicitly set the 'no_data_value' parameter to ensure proper handling of these areas.


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

-    `use_osm_building`: Whether to use OpenStreetMap building data to reduce false positives. Defaults to True

-    `use_osm_roads`: Whether to use OpenStreetMap road data to reduce false positives. Defaults to True

-    `cache_dir`: Directory for storing cached vector data. Defaults to "OWM_cache" in current directory


## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License

## Acknowledgements

Special thanks to the [S1S2-Water dataset authors ](https://github.com/MWieland/s1s2_water) and [The FLAIR #1 dataset authors](https://ignf.github.io/FLAIR/) for providing the valuable training datasets.
