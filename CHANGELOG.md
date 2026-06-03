# Changelog

## [0.5.0] - Jun 3, 2026

### Changed
- **Breaking:** No-data is now written as a GDAL dataset mask via `dst.write_mask()` instead of as a second data band, so GIS software (e.g. QGIS) treats no-data pixels as transparent. The mask is embedded inside the GeoTIFF (`GDAL_TIFF_INTERNAL_MASK`) rather than written as a `.tif.msk` sidecar. Standard output GeoTIFFs now have a single `Water predictions` band; read the mask with `src.read_masks(1)`. Debug output is unchanged.
- Versioning now derives from git tags via `setuptools-scm` (generates `omniwatermask/_version.py` at build time); the hardcoded `omniwatermask/__version__.py` was removed.
- End-to-end tests are now excluded from the default test run (`addopts = "-m 'not e2e'"`); run them explicitly with `pytest -m e2e`.

### Added
- `py.typed` marker and full type hints — the package now ships type information (PEP 561) and is checked with `mypy --strict`.
- Pre-commit hooks (ruff lint/format, mypy, fast tests) and GitHub Actions CI.
- PyPI trove classifiers and project URLs (Repository, Issues, Changelog) in `pyproject.toml`.
- Cloudy Sentinel-2 example notebook demonstrating cloud masking with OmniCloudMask before inference, plus a "Cloudy imagery" guidance section in the README.

## [0.4.3] - Mar 5, 2026

### Fixed
- Fixed unbound variable bug (`NDWI_binary`, `ndwi_target`) when running with `use_ndwi=False` in debug mode
- Fixed mutable default arguments and import-time function calls in `make_water_mask` and `make_water_mask_debug` signatures
- Fixed type hints for `mosaic_device` and `inference_device` to accept `None`
- Fixed exception chaining in `view_cache_db` (`raise ... from e`)

### Added
- Comprehensive pytest test suite (95 tests) covering all modules
- End-to-end tests using real NAIP imagery and real model inference
- NAIP example notebook demonstrating water segmentation on HuggingFace-hosted imagery
- Examples section in README linking to example notebooks
- conda-forge installation instructions in README

### Changed
- Updated OmniCloudMask dependency to v1.7.1 for MPS compatibility
- Moved ruff lint config from deprecated `[tool.ruff]` to `[tool.ruff.lint]`
- Added `strict=True` to `zip()` call in `build_targets` for safety
- Cleaned up docstrings and comments to comply with line length limits

## [0.4.2] - Jan 23, 2026

### Added
- Support for omnicloudmask 1.7
- uv project support with `pyproject.toml` configuration

### Fixed
- Fixed model download location to use packaged CSV instead of models directory
- Temporarily pinned to omnicloudmask v1.6 (later updated to v1.7)

## [0.4.0] - Aug 21, 2025

### Added
- Support for downloading models from Hugging Face using safetensors format
- Added `huggingface_hub` and `safetensors` as dependencies
- Added `destination_model_dir` and `model_download_source` parameters
- Link to published paper in README

## [0.3.0] - May 21, 2025

### Added
- Hugging Face model download support
- No-data mask export as second band in output GeoTIFFs

### Fixed
- Fixed input validation check in debug mode
- Fixed no_data debug output handling

## [0.2.0] - Dec 22, 2024

### Added
- SQLite + Parquet vector caching system for OSM data
- No-data mask export
- Network error handling for OSM requests
- Colab notebook link in README

### Changed
- Refactored target builders for cleaner OSM feature handling

## [0.1.0] - Dec 11, 2024

### Added
- Initial release
- Water segmentation using deep learning model + NDWI + OSM vector data
- Multi-scale threshold optimisation
- Support for multiple sensors (Sentinel-2, Landsat 8, PlanetScope, Maxar, NAIP)
- Configurable band order, patch size, overlap, and inference device
- Vector target building from OpenStreetMap (water, roads, buildings)
- Example notebook with Sentinel-2 mosaic workflow
