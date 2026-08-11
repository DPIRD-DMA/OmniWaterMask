# Changelog

## [0.6.0] - Aug 11, 2026

### Changed
- **Vector data now comes from [Overture Maps](https://overturemaps.org) by default** instead of the Overpass API. Overpass is a live query service that routinely rate-limits or times out on dense urban bounding boxes; Overture serves static monthly GeoParquet releases from cloud storage. On the Sydney example scene, Overpass did not return within 10 minutes, while Overture returned water, road and building vectors in 7–13 s each. Pass `vector_source="osm"` to restore the previous Overpass behaviour.
- Vector cache database bumped to `geodataframes_v2.db`, adding `source` and `ocean` columns so entries from different vector sources cannot be served for one another. An existing v1 cache is ignored rather than migrated, since its rows carry no record of which source produced them.

### Added
- `vector_source` parameter on `make_water_mask` and `make_water_mask_debug`, accepting `"overture"` (default) or `"osm"`.
- `include_ocean` parameter (default True). Overture's `ocean` water features cover everything seaward of the OSM coastline — signal the previous OSM tag set had no equivalent for, since it did not query `natural=coastline`. Set to False where coastline/tide offsets cause false positives.
- `overturemaps>=1.0.0` dependency. `osmnx` is retained for the `vector_source="osm"` path.
- The spaCy exclusion now covers 3.8.15 as well as 3.8.14, and is scoped to Python 3.14 with an environment marker. Both releases publish neither a cp314 wheel nor an sdist, so there is nothing installable on 3.14; 3.8.13 ships both. Other interpreters are no longer held back by the exclusion, and a later release that restores cp314 artifacts will be picked up without another change here.
- The declared `pyarrow` floor is raised from 10.0.0 to 15.0.2. `overturemaps` requires that version, so 10.0.0 was never actually installable alongside it; the old floor only misdescribed what the package supports.

### Fixed
- Overture fetches now retry transient failures (3 attempts, 2s then 4s backoff). Overture is served from S3, where a throttled range read or dropped connection is common and the client does not retry on its own; previously a single blip cost the scene all of its vector targets. A persistent failure now raises rather than returning an empty frame, so a network problem fails the build instead of being read as "no water here".
- A bounding box that genuinely intersects no Overture files (open ocean, Antarctica) is no longer treated as a fetch error. `record_batch_reader` returns `None` for both the empty and the failed case, so the STAC file coverage is checked to tell them apart; if that check is unavailable the `None` is treated as an error, which is the conservative reading.
- Argument validation in `build_targets` moved outside its catch-all `except`, so an invalid `vector_source` raises instead of degrading to a full run with no vector targets and one line in the log.
- A target-building thread that dies without reporting no longer hangs the run. `build_targets` returns its result through a queue, and the reader blocked forever if the thread raised before it could put anything there; the wait now surfaces a `RuntimeError` instead.
- A failed vector build is logged with its traceback (`logging.exception`) rather than just the exception message.
- **A scene whose vector targets fail to build is now skipped instead of exported without them.** Previously the run continued and wrote a mask derived from NDWI and the model alone — not obviously wrong on inspection, and its presence on disk made a later run with `overwrite=False` skip the scene, so one transient outage silently became a permanent result. The scene is now logged at ERROR and left unwritten, and is omitted from the returned output paths; the rest of the batch continues, and a re-run reprocesses it. `build_targets` signals this with a new `TargetBuildError` (put on its queue when threaded, since a raise from a thread would be lost), which is distinct from the `None` it returns when there was nothing to build.
- A skipped scene no longer leaves its other target thread running. Positive and negative targets are built in two threads; propagating the first failure without joining the second left it fetching on into the next scene, so a widespread outage piled orphaned threads up across a batch. Both are now joined before either failure propagates.
- Overpass rate-limiting is no longer invisible. osmnx pauses for its advertised slot time and retries 429/504 after 55s, but reports that through its own logger, which writes nowhere by default — an overloaded Overpass looked like a multi-minute hang. Those messages now route to the standard `logging` module (without turning on osmnx's log files).
- An Overpass response that returns 200 with a body that will not parse as JSON is no longer treated as "no features in this area". osmnx raises `InsufficientResponseError` for both that and a genuinely empty query; the parse failure is now told apart by its chained `JSONDecodeError` and propagates, so a fetch failure cannot be written to the vector cache as an empty result.
- Landforms filed under Overture's water theme are no longer rasterized as positive water targets. Overture's `WaterClass` enum includes `cape` (a headland), `blowhole` (a coastal rock formation) and `shoal` (a routinely exposed sandbank), all carried under `subtype="physical"`. Capes and blowholes are usually points, which `combine_vector_targets` already drops, but shoals do appear as polygons — a Cape Cod bounding box returns two. These would have marked land as water. `natural=cape`/`natural=shoal` were never in `OSM_water_tags`, so dropping them also keeps the two sources aligned. Genuine water in the same subtype (`bay`, `strait`, `sound`) is retained.
- Rasterio dataset handles no longer leak, one or more per scene. The two target threads each read from a dataset opened by the integration layer, and neither that layer nor `export_to_disk` ever closed what it opened, so a long batch could exhaust the process's file descriptors. The handles are now closed in a `finally`, after both threads are joined — closing a dataset a thread is still reading from would take that thread down with it — so they are released whether the scene succeeds, is skipped, or fails during inference.
- An osmnx too old for the Overpass path is now reported as the `ImportError` that says so. The version check ran inside the per-scene target thread, where the raise was lost and the caller saw only "the vector target thread exited without a result", with the real message going to stderr through the threading excepthook rather than the logging module. `make_water_mask`/`make_water_mask_debug` now check up front, before any scene is opened, alongside the existing `vector_source` validation.

### Notes
- Road targets use Overture `segment` features with `subtype="road"`, matching what `highway=*` returned from OSM; rail and waterway segments are excluded.
- Overture building footprints include machine-learning-derived data beyond OSM, so negative building targets have broader coverage than before.
- The Overture path is covered by live network tests marked `e2e`, which assert the fetched schema, subtype vocabulary and land-class filtering against real releases. Run them with `pytest -m e2e`.

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
