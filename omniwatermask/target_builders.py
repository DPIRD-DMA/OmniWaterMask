import logging
from pathlib import Path
from queue import Queue
from typing import Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterio as rio
import torch
from osmnx._errors import InsufficientResponseError
from packaging import version
from pyproj import CRS
from shapely.geometry import box

from .raster_helpers import rasterize_vector
from .vector_cache import add_to_db, check_db

REQUIRED_VERSION = "2.0.0"


def get_osm_features(
    gdf_bounds_4326: gpd.GeoDataFrame,
    tags: dict,
) -> gpd.GeoDataFrame:
    """Download OpenStreetMap features data within a bounding box"""
    gpd_bbox = gdf_bounds_4326.total_bounds
    try:
        features = ox.features_from_bbox(
            bbox=tuple(gpd_bbox),
            tags=tags,
        )
    except InsufficientResponseError:
        logging.info(f"No features found with tags: {tags} within bbox: {gpd_bbox}")
        return gpd.GeoDataFrame()

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
    gdf_bounds = gpd.GeoDataFrame(geometry=[bbox_poly], crs=src.crs)  # type: ignore
    gdf_bounds_4326 = gdf_bounds.to_crs(CRS.from_epsg(4326))
    return gdf_bounds_4326  # type: ignore


def get_aux_data(bbox: gpd.GeoDataFrame, vector_path: Path):
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

    # re-project in 3857 to buffer and then to raster crs
    all_targets = gpd.GeoDataFrame(all_targets).to_crs("EPSG:3857")
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
    ].buffer(distance=5, resolution=8)  # type: ignore

    all_targets["geometry"] = all_targets["geometry"].make_valid()

    all_targets = gpd.GeoDataFrame(all_targets).to_crs(raster_src.crs)

    return gpd.GeoDataFrame(all_targets)


OSM_buildings_tags = {"building": True}
OSM_roads_tags = {"highway": True}
OSM_water_tags = {
    "natural": [
        "water",
        "strait",
    ],
    "waterway": True,
    "water": True,
    "landuse": ["reservoir", "basin"],
    "leisure": ["swimming_pool"],
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
    queue: Queue | None = None,
) -> torch.Tensor | None | Queue:
    """Combine vector for targets into a raster."""
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

        if osm_water or osm_roads or osm_buildings:
            if version.parse(ox.__version__) < version.parse(REQUIRED_VERSION):
                raise ImportError(
                    f"Your installed version of osmnx ({ox.__version__}) is too old. "
                    f"This library requires osmnx version {REQUIRED_VERSION} or above. "
                    f"Please upgrade using 'pip install osmnx>=2.0.0'."
                )

        gdf_bounds_4326 = get_wgs84_bounds_gdf_from_raster(raster_src)

        if use_cache:
            bounds = gdf_bounds_4326.geometry.total_bounds
            polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
            combined_vectors, cache_found = check_db(
                cache_dir=cache_dir,
                polygon=polygon,
                paths=aux_vector_sources,
                water=osm_water,
                roads=osm_roads,
                buildings=osm_buildings,
            )
        else:
            cache_found = False

        if not cache_found:
            all_vectors = []

            for osm_type, tag in zip(
                [osm_water, osm_roads, osm_buildings],
                [OSM_water_tags, OSM_roads_tags, OSM_buildings_tags],
            ):
                if osm_type:
                    response = get_osm_features(
                        gdf_bounds_4326,
                        tags=tag,
                    )
                    if response.empty or response is None:
                        logging.info(f"No {tag} features found")

                    all_vectors.append(response)

            for source in aux_vector_sources:
                logging.info(f"Adding aux vector source: {source.name}")
                all_vectors.append(
                    get_aux_data(bbox=gdf_bounds_4326, vector_path=source)
                )

            combined_vectors = combine_vector_targets(
                vector_list=all_vectors, raster_src=raster_src
            )
            #  add to cache if we are using it, the vectors are not empty, and we did not find a cache
            if use_cache and combined_vectors is not None and not cache_found:
                add_to_db(
                    cache_dir=cache_dir,
                    polygon=polygon,
                    paths=aux_vector_sources,
                    gdf=combined_vectors,
                    water=osm_water,
                    roads=osm_roads,
                    buildings=osm_buildings,
                )
        if combined_vectors is None:
            if queue is not None:
                queue.put(None)
                return queue
            return None

        rasterized_targets = rasterize_vector(
            gdf=combined_vectors, reference_profile=raster_src.profile
        )

        result = torch.from_numpy(rasterized_targets).to(torch.bool).to(device)

        if queue is not None:
            queue.put(result)
            return queue
        return result
    except Exception as e:
        logging.error(f"Error building targets: {e}")
        if queue is not None:
            queue.put(None)
            return queue
        return None
