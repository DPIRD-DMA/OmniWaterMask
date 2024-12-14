import logging
import pickle
from pathlib import Path
from queue import Queue
from typing import Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterio as rio
import torch
from pyproj import CRS
from shapely.geometry import box

from .raster_helpers import rasterize_vector
from .vector_cache import add_to_db, check_db


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
    except Exception as e:
        logging.info(
            f"osmnx failed to locate data with tags: {tags} within bbox: {gpd_bbox}"
        )
        logging.info("This is likely due to no relevant data in the area.")
        logging.info(e)

        return gpd.GeoDataFrame()
    if features.empty:
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


def combine_vector_priors(
    vector_list: list[gpd.GeoDataFrame], raster_src: rio.DatasetReader
) -> Union[gpd.GeoDataFrame, None]:
    logging.info("Combining vector priors")
    all_priors = pd.concat(vector_list, ignore_index=True)

    all_priors.reindex()
    if all_priors.empty:
        return None

    # re-project in 3857 to buffer and then to raster crs
    all_priors = gpd.GeoDataFrame(all_priors).to_crs("EPSG:3857")
    if all_priors is None:
        return None
    # remove points
    all_priors = all_priors[~all_priors.geometry.type.isin(["Point"])]
    logging.info(f"Number of non point features: {len(all_priors)}")

    # get lines and buffer them
    logging.info("Buffering line features")
    line_mask = all_priors.geometry.type.isin(["LineString", "MultiLineString"])

    all_priors.loc[line_mask, "geometry"] = all_priors.loc[
        line_mask, "geometry"
    ].buffer(distance=5, resolution=8)  # type: ignore

    all_priors["geometry"] = all_priors["geometry"].make_valid()

    all_priors = gpd.GeoDataFrame(all_priors).to_crs(raster_src.crs)

    return gpd.GeoDataFrame(all_priors)


def build_priors(
    raster_src: rio.DatasetReader,
    osm_water: bool,
    aux_vector_sources: list[Path],
    device: str,
    cache_dir: Path,
    use_cache: bool = True,
    queue: Queue | None = None,
) -> torch.Tensor | None | Queue:
    """Combine vector water priors into a raster."""
    if not osm_water and not aux_vector_sources:
        logging.info("No water priors to build")
        if queue is not None:
            queue.put(None)
            return queue
        return None

    gdf_bounds_4326 = get_wgs84_bounds_gdf_from_raster(raster_src)

    if use_cache:
        bounds = gdf_bounds_4326.geometry.total_bounds
        polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
        all_water, cache_found = check_db(
            cache_dir=cache_dir,
            polygon=polygon,
            paths=aux_vector_sources,
            water=osm_water,
        )
    else:
        cache_found = False

    if not cache_found:
        skip_cache = False
        all_vectors = []

        if osm_water:
            combined_tags = {
                "natural": [
                    "water",
                    "strait",
                ],
                "waterway": True,
                "water": True,
                "landuse": ["reservoir", "basin"],
                "leisure": ["swimming_pool"],
            }
            response = get_osm_features(
                gdf_bounds_4326,
                tags=combined_tags,
            )
            if response.empty or response is None:
                if use_cache:
                    # no features found dont make cache
                    skip_cache = True
                    logging.info("No water features found so wont make cache")

            all_vectors.append(response)

        for source in aux_vector_sources:
            logging.info(f"Adding aux vector source: {source.name}")
            all_vectors.append(get_aux_data(bbox=gdf_bounds_4326, vector_path=source))

        all_water = combine_vector_priors(
            vector_list=all_vectors, raster_src=raster_src
        )

        if use_cache and not skip_cache and all_water is not None:
            add_to_db(
                cache_dir=cache_dir,
                polygon=polygon,
                paths=aux_vector_sources,
                gdf=all_water,
                water=osm_water,
            )
    if all_water is None:
        if queue is not None:
            queue.put(None)
            return queue
        return None

    rasterized_priors = rasterize_vector(
        gdf=all_water, reference_profile=raster_src.profile
    )

    result = torch.from_numpy(rasterized_priors).to(torch.bool).to(device)

    if queue is not None:
        queue.put(result)
        return queue

    return result


def build_negative_priors(
    raster_src: rio.DatasetReader,
    osm_buildings: bool,
    osm_roads: bool,
    aux_vector_sources: list[Path],
    device: str,
    cache_dir: Path,
    use_cache: bool = True,
    queue: Queue | None = None,
) -> torch.Tensor | None | Queue:
    """Combine vector water priors into a raster."""
    if not osm_buildings and not aux_vector_sources and not osm_roads:
        if queue is not None:
            queue.put(None)
            return queue
        return None

    gdf_bounds_4326 = get_wgs84_bounds_gdf_from_raster(raster_src)
    if use_cache:
        bounds = gdf_bounds_4326.geometry.total_bounds
        polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
        all_negative_priors, cache_found = check_db(
            cache_dir=cache_dir,
            polygon=polygon,
            paths=aux_vector_sources,
            roads=osm_roads,
            buildings=osm_buildings,
        )

    else:
        cache_found = False

    if not cache_found:
        skip_cache = False
        all_vectors = []

        if osm_buildings:
            response = get_osm_features(gdf_bounds_4326, tags={"building": True})
            all_vectors.append(response)
            if response.empty or response is None:
                # no features found dont make cache
                skip_cache = True
                logging.info("No building features found so wont make cache")

        if osm_roads:
            response = get_osm_features(gdf_bounds_4326, tags={"highway": True})
            all_vectors.append(response)
            if response.empty or response is None:
                if use_cache:
                    # no features found dont make cache
                    skip_cache = True
                    logging.info("No road features found so wont make cache")
        for source in aux_vector_sources:
            all_vectors.append(get_aux_data(bbox=gdf_bounds_4326, vector_path=source))

        all_negative_priors = combine_vector_priors(
            vector_list=all_vectors, raster_src=raster_src
        )

        if use_cache and not skip_cache and all_negative_priors is not None:
            add_to_db(
                cache_dir=cache_dir,
                polygon=polygon,
                paths=aux_vector_sources,
                gdf=all_negative_priors,
                roads=osm_roads,
                buildings=osm_buildings,
            )

    if all_negative_priors is None:
        if queue is not None:
            queue.put(None)
            return queue
        return None

    rasterized_negative_priors = rasterize_vector(
        gdf=all_negative_priors, reference_profile=raster_src.profile
    )
    result = torch.from_numpy(rasterized_negative_priors).to(torch.bool).to(device)

    if queue is not None:
        queue.put(result)
        return queue

    return result
