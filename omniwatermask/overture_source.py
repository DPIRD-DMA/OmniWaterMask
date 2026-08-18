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
import threading
import time
from typing import Any, Optional

import geopandas as gpd
import pyarrow.fs as fs
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

# Overture publishes release discovery through a STAC catalogue, and
# ``overturemaps`` resolves ``release=None`` through it on every call - before
# it consults ``stac``, so ``stac=False`` does not avoid it. A single missing
# object there took every uncached scene down in August 2026 while the Parquet
# itself stayed readable on S3, so the release is resolved here instead, once
# per process, with S3 as the fallback.
_release_lock = threading.Lock()
_resolved_release: Optional[str] = None

# The bucket the releases live in. Listing it is the fallback for discovery,
# and needs no credentials.
RELEASE_BUCKET = "overturemaps-us-west-2"
RELEASE_PREFIX = f"{RELEASE_BUCKET}/release"

# A release directory can appear on S3 before it is fully uploaded, and a bare
# listing carries no "published" signal the way the catalogue's ``latest`` key
# does. Requiring the themes OWM actually reads keeps a half-written release
# from being chosen over an older one that has them.
KNOWN_REQUIRED_THEMES = frozenset(
    {"theme=base", "theme=transportation", "theme=buildings"}
)


def _required_themes(type_theme_map: dict[str, str]) -> frozenset[str]:
    """Derive required themes, tolerating an incomplete upstream mapping.

    Derived from the types OWM asks for, so adding a kind to ``OVERTURE_TYPES``
    cannot leave this behind. ``type_theme_map`` is public but not guaranteed,
    so an incomplete one falls back to the themes those types map to today.
    """
    try:
        return frozenset(
            f"theme={type_theme_map[overture_type]}"
            for overture_type in OVERTURE_TYPES.values()
        )
    except KeyError as e:
        logging.warning(
            "Could not derive required Overture themes from overturemaps (%s); "
            "using the known theme set instead.",
            e,
        )
        return KNOWN_REQUIRED_THEMES


try:
    _TYPE_THEME_MAP = core.type_theme_map
except AttributeError as e:  # pragma: no cover - upstream dropped the mapping
    logging.warning(
        "Could not derive required Overture themes from overturemaps (%s); "
        "using the known theme set instead.",
        e,
    )
    REQUIRED_THEMES = KNOWN_REQUIRED_THEMES
else:
    REQUIRED_THEMES = _required_themes(_TYPE_THEME_MAP)


def _release_sort_key(release: str) -> tuple[str, int]:
    """Sort ``YYYY-MM-DD.sequence`` releases by numeric sequence."""
    date, separator, sequence = release.rpartition(".")
    if separator and sequence.isdigit():
        return date, int(sequence)
    return release, -1


def _latest_release_from_s3() -> str:
    """Return the newest release on S3 that carries every theme OWM reads.

    Checks that the theme directories are present, which is not the same as
    checking that their contents finished uploading - it is the cheap guard
    available from a listing, not a completeness proof.
    """
    filesystem = fs.S3FileSystem(anonymous=True, region="us-west-2")
    selector = fs.FileSelector(RELEASE_PREFIX, recursive=False)
    releases = sorted(
        (
            info.base_name
            for info in filesystem.get_file_info(selector)
            if info.type == fs.FileType.Directory
        ),
        key=_release_sort_key,
        reverse=True,
    )

    for release in releases:
        themes = {
            info.base_name
            for info in filesystem.get_file_info(
                fs.FileSelector(f"{RELEASE_PREFIX}/{release}", recursive=False)
            )
        }
        if REQUIRED_THEMES <= themes:
            return str(release)
        logging.debug(f"Skipping incomplete Overture release {release}: {themes}")

    raise RuntimeError(
        f"No usable Overture release found under s3://{RELEASE_PREFIX}. "
        f"Found {releases or 'none'}, none carrying {sorted(REQUIRED_THEMES)}."
    )


def _invalidate_release(release: str) -> None:
    """Forget ``release`` so the next fetch rediscovers one.

    Overture prunes releases roughly monthly. A long-lived process - a
    notebook, a worker, a service - would otherwise keep asking for a release
    that has since been deleted, and every later fetch would fail against it.

    Only clears the cache if it still holds ``release``: a slow fetch failing
    against a pruned release must not discard a newer one that another thread
    has since resolved.
    """
    global _resolved_release

    with _release_lock:
        if _resolved_release == release:
            _resolved_release = None


def _resolve_release() -> str:
    """Resolve the Overture release to read, once per process.

    Tries Overture's own discovery first so a healthy catalogue still decides,
    and falls back to the newest release on S3 carrying every theme OWM reads
    when it is unavailable.
    Deliberately not a pinned default: Overture keeps roughly two releases
    (~60 days) and prunes the rest, so any hardcoded date stops resolving
    within a couple of months.
    """
    global _resolved_release

    with _release_lock:
        if _resolved_release is not None:
            return _resolved_release

        try:
            latest = core.get_latest_release()
            if not latest:
                # A reachable catalogue whose "latest" key is missing or null
                # returns None rather than raising. Without this it would be
                # str()-ed into the literal "None" and read as a release id.
                raise RuntimeError("Overture's catalogue names no latest release")
            release = str(latest)
        except Exception as e:
            logging.warning(
                f"Overture release discovery is unavailable ({e}). Falling back "
                f"to the newest release under s3://{RELEASE_PREFIX} carrying "
                f"{sorted(REQUIRED_THEMES)}."
            )
            try:
                release = _latest_release_from_s3()
            except Exception as fallback_error:
                raise RuntimeError(
                    "Could not determine an Overture release: discovery failed "
                    f"({e}) and the S3 fallback failed ({fallback_error}). If "
                    "Overture is unavailable, OpenStreetMap is an alternative "
                    'source - pass vector_source="osm" to query it instead.'
                ) from fallback_error

        logging.info(f"Using Overture release {release}")
        _resolved_release = release
        return release


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
    release: str,
) -> Optional[gpd.GeoDataFrame]:
    """Run a single fetch attempt, or return None if Overture gave no reader.

    ``record_batch_reader`` streams lazily, so a failed range read surfaces out
    of ``from_arrow`` rather than out of the reader call. Both live here so the
    retry above covers the whole request.
    """
    # stac=True targets only the files intersecting the bbox via the STAC file
    # index. Without it the whole dataset listing is opened, which measured
    # ~5x slower per request. ``release`` is passed explicitly because the
    # client otherwise resolves it per call through the catalogue root, which
    # is a separate service from the per-release index this relies on.
    reader = core.record_batch_reader(
        overture_type, bbox=bbox, release=release, stac=True
    )
    if reader is None:
        return None
    return gpd.GeoDataFrame.from_arrow(reader)


def _bbox_has_no_files(
    overture_type: str,
    bbox: tuple[float, float, float, float],
    release: str,
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
        return (
            prepare_query(overture_type, bbox=bbox, release=release, stac=True) is None
        )
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
    # Resolved outside the loop: which release to read is a property of the run,
    # not of this bbox, and a discovery failure is deterministic. Retrying it
    # would spend the whole backoff schedule re-asking a question that has
    # already been answered.
    release = _resolve_release()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            features = _fetch_once(overture_type, bbox, release)
        except Exception as e:
            last_error = e
        else:
            if features is not None:
                return features
            if _bbox_has_no_files(overture_type, bbox, release):
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

    # The release may have been pruned out from under a long-lived process, so
    # drop it and let the next fetch resolve the current one.
    _invalidate_release(release)

    raise RuntimeError(
        f"Overture {kind} fetch failed after {MAX_ATTEMPTS} attempts for bbox: "
        f"{bbox} (release {release}). This usually indicates a network or cloud "
        "storage error. If Overture stays unavailable, OpenStreetMap covers the "
        'same water and road data - pass vector_source="osm" to query it '
        "through the Overpass API instead."
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
