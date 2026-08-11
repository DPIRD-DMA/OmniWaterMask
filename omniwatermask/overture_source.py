"""Fetch vector features from Overture Maps.

Overture is an alternative to the Overpass API that ``osmnx`` queries. Overpass
is a live query service that routinely rate-limits or times out on dense urban
bounding boxes; Overture serves static, monthly-released GeoParquet from cloud
storage, so a fetch is a bounded range-read rather than a server-side query.

The underlying data is largely the same: Overture's ``base`` and
``transportation`` themes are derived from OpenStreetMap. The ``buildings``
theme adds machine-learning-derived footprints on top of OSM.
"""

import logging
import time
from typing import Any, Optional

import geopandas as gpd
from overturemaps import core

# Overture type names for each kind of feature OWM targets, chosen to match the
# OSM tag sets in target_builders: OSM_water_tags -> water, OSM_roads_tags
# (highway=*) -> road segments, OSM_buildings_tags -> buildings.
OVERTURE_TYPES: dict[str, str] = {
    "water": "water",
    "roads": "segment",
    "buildings": "building",
}

# The transportation theme also carries "rail" and "water" segments. Only
# "road" is kept so negative targets match what highway=* returned from OSM.
OVERTURE_SUBTYPES: dict[str, frozenset[str]] = {
    "roads": frozenset({"road"}),
}

# Everything seaward of the OSM coastline is a single "ocean" water feature.
# OSM_water_tags has no coastline equivalent, so this is signal OWM did not
# previously receive - but it is pinned to the coastline vector regardless of
# tide, hence the opt-out in build_targets.
OCEAN_SUBTYPE = "ocean"

# Overture files a few landforms under the water theme's "physical" subtype:
# a cape is a headland, a blowhole is a coastal rock formation, and a shoal is
# a sandbank that is routinely exposed. Rasterizing them as positive water
# targets marks land as water. Capes and blowholes are usually points (which
# combine_vector_targets drops anyway), but shoals do appear as polygons.
# OSM_water_tags never selected these - it queried natural=water/strait, not
# natural=cape/shoal - so dropping them also keeps the two sources aligned.
OVERTURE_LAND_CLASSES = frozenset({"cape", "blowhole", "shoal"})

# Only these are needed downstream: combine_vector_targets uses geometry alone,
# and the subtype/class pair is kept for filtering, cache inspection and
# debugging.
# Dropping the rest keeps the concat in combine_vector_targets and the parquet
# cache free of Overture's deeply nested attribute columns.
KEEP_COLUMNS = ["subtype", "class", "geometry"]

# Overture is served from S3, so failures are transient far more often than not:
# a throttled range read, a dropped connection, a timeout on a dense bbox. The
# client itself does not retry, so a single blip would otherwise cost the scene
# all of its vector targets. Delays are 2s, 4s - short enough not to stall a
# batch run, long enough to clear a throttle.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def get_overture_features(
    gdf_bounds_4326: gpd.GeoDataFrame,
    kind: str,
    include_ocean: bool = True,
) -> gpd.GeoDataFrame:
    """Download Overture features of ``kind`` within a bounding box.

    ``kind`` is one of ``water``, ``roads`` or ``buildings``. ``include_ocean``
    only applies to ``water``.
    """
    if kind not in OVERTURE_TYPES:
        raise ValueError(
            f"Unknown Overture feature kind: {kind!r}. "
            f"Expected one of {sorted(OVERTURE_TYPES)}."
        )
    overture_type = OVERTURE_TYPES[kind]
    bounds = gdf_bounds_4326.total_bounds
    bbox = (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )

    features = _fetch_with_retry(overture_type=overture_type, bbox=bbox, kind=kind)

    if features.empty:
        logging.info(f"No {kind} features found within bbox: {bbox}")
        return gpd.GeoDataFrame()

    features = _filter_features(features, kind=kind, include_ocean=include_ocean)
    if features.empty:
        logging.info(f"No {kind} features left after filtering")
        return gpd.GeoDataFrame()

    features = features[[c for c in KEEP_COLUMNS if c in features.columns]]
    features = features.set_crs("EPSG:4326", allow_override=True)
    features = gpd.clip(features, gdf_bounds_4326)
    return gpd.GeoDataFrame(features)


def _fetch_once(
    overture_type: str,
    bbox: tuple[float, float, float, float],
) -> Optional[gpd.GeoDataFrame]:
    """Run a single fetch attempt, or return None if Overture gave no reader.

    ``record_batch_reader`` streams lazily, so a failed range read surfaces out
    of ``from_arrow`` rather than out of the reader call. Both live here so the
    retry above covers the whole request.
    """
    # stac=True lets the client resolve the release and target only the
    # relevant files via the STAC catalogue. Without it the whole dataset
    # listing is opened, which measured ~5x slower per request.
    reader = core.record_batch_reader(overture_type, bbox=bbox, stac=True)
    if reader is None:
        return None
    return gpd.GeoDataFrame.from_arrow(reader)


def _bbox_has_no_files(
    overture_type: str,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Report whether STAC finds no files at all intersecting ``bbox``.

    ``record_batch_reader`` returns None both when the bbox is genuinely empty
    (STAC matched zero files, e.g. open ocean or Antarctica) and when opening
    the dataset failed. Only the second is an error, and the two are
    indistinguishable from the None alone. ``_prepare_query`` is what returns
    None for the empty case, so ask it directly. It is private, so an
    overturemaps release that drops or renames it makes this report False and
    the caller treats the None as an error - the conservative reading.
    """
    prepare_query = getattr(core, "_prepare_query", None)
    if prepare_query is None:
        return False
    try:
        return prepare_query(overture_type, bbox=bbox, stac=True) is None
    except Exception as e:
        logging.debug(f"Could not check STAC file coverage for {overture_type}: {e}")
        return False


def _fetch_with_retry(
    overture_type: str,
    bbox: tuple[float, float, float, float],
    kind: str,
) -> gpd.GeoDataFrame:
    """Fetch ``overture_type`` over ``bbox``, retrying transient failures.

    Returns an empty frame when the bbox genuinely holds no features. Raises
    once the attempts are exhausted, so a persistent failure fails the build
    rather than passing an empty frame off as "no water here".
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            features = _fetch_once(overture_type, bbox)
        except Exception as e:
            last_error = e
        else:
            if features is not None:
                return features
            if _bbox_has_no_files(overture_type, bbox):
                logging.info(f"Overture has no {kind} files intersecting bbox: {bbox}")
                return gpd.GeoDataFrame()
            last_error = RuntimeError(
                f"Overture returned no reader for type {overture_type!r} "
                f"within bbox: {bbox}"
            )

        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            logging.warning(
                f"Overture {kind} fetch failed (attempt {attempt}/{MAX_ATTEMPTS}): "
                f"{last_error}. Retrying in {delay:.0f}s"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Overture {kind} fetch failed after {MAX_ATTEMPTS} attempts for bbox: "
        f"{bbox}. This usually indicates a network or cloud storage error."
    ) from last_error


def _filter_features(
    features: gpd.GeoDataFrame,
    kind: str,
    include_ocean: bool,
) -> gpd.GeoDataFrame:
    """Restrict features to those matching the equivalent OSM tag sets."""
    if kind == "water" and "class" in features.columns:
        features = features[~features["class"].isin(OVERTURE_LAND_CLASSES)]

    if "subtype" not in features.columns:
        return gpd.GeoDataFrame(features)

    keep: Any = OVERTURE_SUBTYPES.get(kind)
    if keep is not None:
        features = features[features["subtype"].isin(keep)]

    if kind == "water" and not include_ocean:
        features = features[features["subtype"] != OCEAN_SUBTYPE]

    return gpd.GeoDataFrame(features)
