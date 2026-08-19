import logging
from pathlib import Path
from queue import Queue
from typing import Any, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterio as rio
import torch
from osmnx._errors import InsufficientResponseError
from packaging import version
from pyproj import CRS
from shapely.geometry import box

from .overture_source import get_overture_features
from .raster_helpers import rasterize_vector
from .vector_cache import add_to_db, check_db

REQUIRED_VERSION = "2.0.0"

OVERTURE = "overture"
OSM = "osm"
VECTOR_SOURCES = (OVERTURE, OSM)


class TargetBuildError(RuntimeError):
    """A scene's vector targets could not be built.

    Raised in place of returning targets, so a scene is skipped rather than
    exported from NDWI and the model alone. A mask built without its vector
    targets is not obviously wrong on inspection, so it must not be written.
    """


def validate_vector_source(vector_source: str) -> None:
    """Raise if ``vector_source`` is not one this library knows about."""
    if vector_source not in VECTOR_SOURCES:
        raise ValueError(
            f"Unknown vector_source: {vector_source!r}. "
            f"Expected one of {list(VECTOR_SOURCES)}."
        )


def check_osmnx_version() -> None:
    """Raise if the installed osmnx is too old for the Overpass path."""
    if version.parse(ox.__version__) < version.parse(REQUIRED_VERSION):
        raise ImportError(
            f"Your installed version of osmnx ({ox.__version__}) is too old. "
            f"This library requires osmnx version {REQUIRED_VERSION} or above. "
            f"Please upgrade using 'pip install osmnx>=2.0.0'."
        )


def _route_osmnx_logging() -> None:
    """Send osmnx's own log messages to the standard logging module.

    Overpass rate-limiting is handled inside osmnx, which pauses for its
    advertised slot time and retries 429/504 responses after 55s. It reports all
    of that through ``osmnx.utils.log``, which by default writes nowhere - so an
    overloaded Overpass looks like a multi-minute hang with no output.

    ``utils.log`` only reaches a logger when ``settings.log_file`` is set, and
    ``_get_logger`` attaches a file handler of its own unless the logger already
    has one. Adding a NullHandler first claims that slot, so the messages
    propagate to the root logger and OWM's configuration instead of to a file
    under ``./logs``. The level is left unset so it inherits whatever the caller
    configured; the 429/504 retry notice is logged at WARNING and shows up even
    under Python's default configuration.
    """
    osmnx_logger = logging.getLogger(ox.settings.log_name)
    if not osmnx_logger.handlers:
        osmnx_logger.addHandler(logging.NullHandler())
    osmnx_logger.propagate = True
    ox.settings.log_file = True


def get_osm_features(
    gdf_bounds_4326: gpd.GeoDataFrame,
    tags: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Download OpenStreetMap features data within a bounding box"""
    _route_osmnx_logging()
    gpd_bbox = gdf_bounds_4326.total_bounds
    try:
        features = ox.features_from_bbox(
            bbox=(
                float(gpd_bbox[0]),
                float(gpd_bbox[1]),
                float(gpd_bbox[2]),
                float(gpd_bbox[3]),
            ),
            tags=tags,
        )
    except InsufficientResponseError as e:
        # osmnx raises this both for a query that genuinely matched nothing and
        # for a 200 response whose body would not parse as JSON. Only the first
        # is an empty area; swallowing the second would cache a fetch failure as
        # "no features here". The parse path chains the JSONDecodeError that
        # caused it, so a cause means it was not an empty area.
        if e.__cause__ is not None:
            raise RuntimeError(
                f"Overpass returned an unreadable response for tags: {tags} "
                f"within bbox: {gpd_bbox}. Overpass rate-limits and times out on "
                "large or dense areas; if that is what happened, Overture serves "
                "the same underlying water and road data as static files from "
                'cloud storage - pass vector_source="overture" to read it '
                "instead."
            ) from e
        logging.info(f"No features found with tags: {tags} within bbox: {gpd_bbox}")
        return gpd.GeoDataFrame()
    except Exception as e:
        # Deliberately broad: anything that is not an empty area must not be
        # cached as "no features here". The cause is chained, so the real error
        # survives - but it may be a malformed tags dict or an osmnx bug rather
        # than Overpass being unavailable, so the alternative source is offered
        # conditionally rather than asserted as the diagnosis.
        raise RuntimeError(
            f"OpenStreetMap fetch failed for tags: {tags} within bbox: "
            f"{gpd_bbox}. See the chained exception for the cause. If Overpass "
            "is unavailable or rate-limiting, Overture serves the same "
            "underlying water and road data as static files from cloud storage "
            '- pass vector_source="overture" to read it instead.'
        ) from e

    features = features.drop(columns=["nodes", "ways"], errors="ignore")
    features = features.to_crs("EPSG:4326")
    features = gpd.clip(features, gdf_bounds_4326)
    return gpd.GeoDataFrame(features)


def get_wgs84_bounds_gdf_from_raster(
    src: rio.DatasetReader,
) -> gpd.GeoDataFrame:
    """Get the bounds of a raster in WGS84"""
    bounds = src.bounds
    bbox_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gdf_bounds = gpd.GeoDataFrame(geometry=[bbox_poly], crs=src.crs)
    gdf_bounds_4326 = gdf_bounds.to_crs(CRS.from_epsg(4326))
    return gdf_bounds_4326


def get_aux_data(bbox: gpd.GeoDataFrame, vector_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(vector_path, bbox=bbox)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def combine_vector_targets(
    vector_list: list[gpd.GeoDataFrame], raster_src: rio.DatasetReader
) -> Union[gpd.GeoDataFrame, None]:
    logging.info("Combining vector targets")
    all_targets = pd.concat(vector_list, ignore_index=True)

    all_targets.reindex()
    if all_targets.empty:
        return None

    # Buffering needs a projected CRS so the distance is in metres, and the
    # raster's own CRS normally is one - Sentinel-2 is UTM. Reprojecting
    # straight to it does in one pass what used to take two, a round trip out
    # to EPSG:3857 and back, which cost 2.1 GB of the 2.7 GB this function
    # spent on a Perth tile's 918k features.
    #
    # It also buffers by the distance actually asked for. EPSG:3857's scale
    # factor is 1/cos(latitude), so buffering 5 m there lays down ~4.2 m on the
    # ground at Perth, and a different amount at every other latitude. Only a
    # geographic raster CRS still needs the detour.
    raster_crs = CRS.from_user_input(raster_src.crs)
    buffer_crs = raster_crs if raster_crs.is_projected else CRS.from_epsg(3857)

    all_targets = gpd.GeoDataFrame(all_targets).to_crs(buffer_crs)
    if all_targets is None:
        return None
    # remove points
    all_targets = all_targets[~all_targets.geometry.type.isin(["Point"])]
    logging.info(f"Number of non point features: {len(all_targets)}")

    # get lines and buffer them
    logging.info("Buffering line features")
    line_mask = all_targets.geometry.type.isin(["LineString", "MultiLineString"])

    all_targets.loc[line_mask, "geometry"] = all_targets.loc[
        line_mask, "geometry"
    ].buffer(distance=5, resolution=8)

    all_targets["geometry"] = all_targets.geometry.make_valid()

    if buffer_crs != raster_crs:
        all_targets = gpd.GeoDataFrame(all_targets).to_crs(raster_crs)

    return gpd.GeoDataFrame(all_targets)


OSM_buildings_tags: dict[str, Any] = {"building": True}
OSM_roads_tags: dict[str, Any] = {"highway": True}
OSM_water_tags: dict[str, Any] = {
    "natural": [
        "water",
        "strait",
    ],
    "waterway": True,
    "water": True,
    "landuse": ["reservoir", "basin"],
    "leisure": ["swimming_pool"],
}

OSM_tags_by_kind: dict[str, dict[str, Any]] = {
    "water": OSM_water_tags,
    "roads": OSM_roads_tags,
    "buildings": OSM_buildings_tags,
}


def build_targets(
    raster_src: rio.DatasetReader,
    aux_vector_sources: list[Path],
    device: str,
    cache_dir: Path,
    osm_water: bool = False,
    osm_roads: bool = False,
    osm_buildings: bool = False,
    use_cache: bool = True,
    all_touched: bool = False,
    vector_source: str = OVERTURE,
    include_ocean: bool = True,
    queue: Queue[Any] | None = None,
) -> torch.Tensor | None | Queue[Any]:
    """Combine vector for targets into a raster.

    ``all_touched`` controls the rasterization rule. ``False`` (centre-based)
    only flips a pixel when a feature covers roughly the majority of it, which
    suits positive water targets where small sub-pixel features should not turn
    a whole pixel on. ``True`` flips a pixel if any feature touches it at all,
    which suits negative targets (buildings/roads) where we want to aggressively
    exclude anything those features clip.

    ``vector_source`` selects where the water/road/building vectors come from:
    ``"overture"`` reads Overture Maps GeoParquet from cloud storage, ``"osm"``
    queries the Overpass API through ``osmnx``. ``include_ocean`` only applies
    to Overture water targets.

    Returns None when there was nothing to build. A build that was asked for and
    failed raises ``TargetBuildError`` instead, so the caller can tell the two
    apart. When ``queue`` is given the error is put on it rather than raised,
    since this runs as a thread target where a raise would be lost; the reader
    is expected to re-raise it.
    """
    # Argument validation sits outside the try below so a misconfigured call
    # fails immediately. Inside it, the catch-all would turn a typo into a full
    # run with no vector targets and a single line in the log.
    validate_vector_source(vector_source)

    if (osm_water or osm_roads or osm_buildings) and vector_source == OSM:
        check_osmnx_version()

    # Ocean only ever affects water, so builds without it record ocean as
    # off. Otherwise toggling the flag would needlessly invalidate cached
    # building/road targets it cannot change.
    include_ocean = include_ocean and osm_water and vector_source == OVERTURE

    try:
        if (
            not osm_water
            and not osm_roads
            and not osm_buildings
            and not aux_vector_sources
        ):
            logging.info("No water targets to build")
            if queue is not None:
                queue.put(None)
                return queue
            return None

        gdf_bounds_4326 = get_wgs84_bounds_gdf_from_raster(raster_src)

        bounds = gdf_bounds_4326.geometry.total_bounds
        polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])

        combined_vectors = None
        if use_cache:
            combined_vectors, cache_found = check_db(
                cache_dir=cache_dir,
                polygon=polygon,
                paths=aux_vector_sources,
                water=osm_water,
                roads=osm_roads,
                buildings=osm_buildings,
                source=vector_source,
                ocean=include_ocean,
            )
        else:
            cache_found = False

        if not cache_found:
            all_vectors = []

            for kind, enabled in zip(
                ["water", "roads", "buildings"],
                [osm_water, osm_roads, osm_buildings],
                strict=True,
            ):
                if enabled:
                    if vector_source == OSM:
                        response = get_osm_features(
                            gdf_bounds_4326,
                            tags=OSM_tags_by_kind[kind],
                        )
                    else:
                        response = get_overture_features(
                            gdf_bounds_4326,
                            kind=kind,
                            include_ocean=include_ocean,
                        )
                    if response is None or response.empty:
                        logging.info(f"No {kind} features found")

                    all_vectors.append(response)

            for source in aux_vector_sources:
                logging.info(f"Adding aux vector source: {source.name}")
                all_vectors.append(
                    get_aux_data(bbox=gdf_bounds_4326, vector_path=source)
                )

            combined_vectors = combine_vector_targets(
                vector_list=all_vectors, raster_src=raster_src
            )
            #  add to cache if using it, vectors are not empty, and no cache found
            if use_cache and combined_vectors is not None and not cache_found:
                add_to_db(
                    cache_dir=cache_dir,
                    polygon=polygon,
                    paths=aux_vector_sources,
                    gdf=combined_vectors,
                    water=osm_water,
                    roads=osm_roads,
                    buildings=osm_buildings,
                    source=vector_source,
                    ocean=include_ocean,
                )
        if combined_vectors is None:
            if queue is not None:
                queue.put(None)
                return queue
            return None

        rasterized_targets = rasterize_vector(
            gdf=combined_vectors,
            reference_profile=raster_src.profile,
            all_touched=all_touched,
        )

        result = torch.from_numpy(rasterized_targets).to(torch.bool).to(device)

        if queue is not None:
            queue.put(result)
            return queue
        return result
    except Exception as e:
        logging.exception("Error building targets")
        error = TargetBuildError(f"Could not build vector targets: {e}")
        # Raising from a thread would be lost, so the queue carries the error
        # for the reader to re-raise. Chain it by hand: the raise below is what
        # normally sets __cause__, and the queued copy needs it too.
        if queue is not None:
            error.__cause__ = e
            queue.put(error)
            return queue
        raise error from e
