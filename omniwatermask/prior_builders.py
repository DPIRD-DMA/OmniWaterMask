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
from shapely import prepare
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .raster_helpers import rasterize_vector


def optimized_remove_polygon_holes(
    outer_polygons: list[Polygon],
    inner_polygons: list[Polygon],
) -> Polygon | MultiPolygon:
    """The existing OSMnx function is slow when removing holes from polygons.
    Fix in progress:
    https://github.com/gboeing/osmnx/issues/1200
    """
    if len(inner_polygons) == 0:
        # if there are no holes to remove, geom is the union of outer polygons
        geometry = unary_union(outer_polygons)
    else:
        # otherwise, remove from each outer poly each inner poly it contains
        polygons_with_holes = []
        for outer in outer_polygons:
            prepare(outer)
            holes = [inner for inner in inner_polygons if outer.contains(inner)]
            polygons_with_holes.append(outer.difference(unary_union(holes)))
        geometry = unary_union(polygons_with_holes)

    # ensure returned geometry is a Polygon or MultiPolygon
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    return Polygon()


# Monkey patch the OSMnx function
ox.features._remove_polygon_holes = optimized_remove_polygon_holes


def get_osm_features(
    gdf_bounds_4326: gpd.GeoDataFrame,
    tags: dict,
) -> gpd.GeoDataFrame:
    """Download OpenStreetMap features data within a bounding box"""
    gpd_bbox = gdf_bounds_4326.total_bounds
    try:
        features = ox.features_from_bbox(
            bbox=(gpd_bbox[3], gpd_bbox[1], gpd_bbox[0], gpd_bbox[2]),
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
    # osm_water = osm_water.clip(gdf_bounds_4326)
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

    all_priors.loc[line_mask, "geometry"] = all_priors.loc[line_mask, "geometry"].buffer(distance=5, resolution=8)  # type: ignore

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
    aux_vector_names = ""
    for aux_vector in aux_vector_sources:
        aux_vector_names += aux_vector.stem + "_"

    file_name = f"water_priors_{gdf_bounds_4326.to_string()}_{osm_water}_{aux_vector_names}.pkl".replace(
        " ", ""
    )
    cache_path = cache_dir / "water_priors_cache" / file_name
    logging.info(f"Cache path: {cache_path}")
    skip_cache = False

    if use_cache and cache_path.exists():
        logging.info(f"Loading water priors from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_water = pickle.load(f)
        # all_water = pickle.load(open(cache_path, "rb"))
    else:
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

        if use_cache and not skip_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_water, f)
                # pickle.dump(all_water, open(cache_path, "wb"))
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
    aux_vector_names = ""
    for aux_vector in aux_vector_sources:
        aux_vector_names += aux_vector.stem + "_"
    file_name = f"negative_priors_{gdf_bounds_4326.to_string()}_{osm_buildings}_{aux_vector_names}_{osm_roads}.pkl".replace(
        " ", ""
    )
    cache_path = cache_dir / "negative_priors_cache" / file_name
    skip_cache = False

    if use_cache and cache_path.exists():
        logging.info(f"Loading negative priors from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_negative_priors = pickle.load(f)
        # all_negative_priors = pickle.load(open(cache_path, "rb"))

    else:
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

        if use_cache and not skip_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_negative_priors, f)

            # pickle.dump(all_negative_priors, open(cache_path, "wb"))

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
