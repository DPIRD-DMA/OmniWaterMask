import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

DB_NAME = "geodataframes_v1.db"
GDF_DIR = "gdfs"


def initialize_db(cache_dir: Path) -> None:
    """
    Initializes the SQLite database in the given cache directory.
    """
    db_path = cache_dir / DB_NAME
    gdf_dir = cache_dir / GDF_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS geodataframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                polygon TEXT NOT NULL,
                paths TEXT NOT NULL,
                water BOOLEAN NOT NULL,
                roads BOOLEAN NOT NULL,
                buildings BOOLEAN NOT NULL,
                gdf_uid TEXT NOT NULL
            )
        """)
        conn.commit()


def check_db(
    cache_dir: Path,
    polygon: Polygon,
    paths: List[Path],
    water: bool = False,
    roads: bool = False,
    buildings: bool = False,
) -> tuple[gpd.GeoDataFrame, bool]:
    """
    Checks the database for an existing GeoDataFrame matching the given parameters.

    Args:
        cache_dir: Directory where the database is stored.
        polygon: Polygon geometry to check.
        paths: List of file paths.
        water: Boolean flag for water.
        roads: Boolean flag for roads.
        buildings: Boolean flag for buildings.

    Returns:
        The matching GeoDataFrame if found, otherwise False.
    """
    db_path = cache_dir / DB_NAME
    gdf_dir = cache_dir / GDF_DIR
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT gdf_uid FROM geodataframes
            WHERE polygon = ? AND paths = ? AND water = ? AND roads = ? AND buildings = ?
        """,
            (polygon.wkt, json.dumps([str(p) for p in paths]), water, roads, buildings),
        )
        row = cursor.fetchone()
        if row:
            logging.info("Found matching GeoDataFrame in cache")
            gdf_path = gdf_dir / f"{row[0]}.parquet"
            logging.info(f"Loading GeoDataFrame from {gdf_path}")
            if gdf_path.exists():
                return gpd.read_parquet(gdf_path), True  # Load the GDF from disk
            else:
                logging.warning(f"GeoDataFrame file {gdf_path} not found")
        logging.info("No matching GeoDataFrame found in cache")
        return gpd.GeoDataFrame(), False


def add_to_db(
    cache_dir: Path,
    polygon: Polygon,
    paths: List[Path],
    gdf: gpd.GeoDataFrame,
    water: bool = False,
    roads: bool = False,
    buildings: bool = False,
) -> None:
    """
    Adds a new GeoDataFrame entry to the database.

    Args:
        cache_dir: Directory where the database is stored.
        polygon: Polygon geometry to store.
        paths: List of file paths.
        water: Boolean flag for water.
        roads: Boolean flag for roads.
        buildings: Boolean flag for buildings.
        gdf: The GeoDataFrame to store.
    """
    db_path = cache_dir / DB_NAME
    gdf_dir = cache_dir / GDF_DIR
    gdf_uid = str(uuid.uuid4())  # Generate a unique identifier
    gdf_file = gdf_dir / f"{gdf_uid}.parquet"  # File name based on UID
    gdf.to_parquet(gdf_file)  # Save the GDF to disk in Parquet format

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO geodataframes (polygon, paths, water, roads, buildings, gdf_uid)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                polygon.wkt,
                json.dumps([str(p) for p in paths]),
                water,
                roads,
                buildings,
                gdf_uid,
            ),
        )
        conn.commit()
        logging.info("Added GeoDataFrame to cache")


def view_cache_db(cache_dir: Path) -> pd.DataFrame:
    """
    Views the contents of the geodataframes database as a pandas DataFrame.

    Args:
        cache_dir: Directory where the database is stored

    Returns:
        pandas.DataFrame containing all records from the geodataframes table

    """
    db_path = cache_dir / DB_NAME

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    try:
        with sqlite3.connect(db_path) as conn:
            # Read the entire table into a pandas DataFrame
            query = "SELECT * FROM geodataframes"
            df = pd.read_sql_query(query, conn)

            # Convert the JSON string of paths back to a list
            import json

            df["paths"] = df["paths"].apply(json.loads)

            # Convert boolean integers to actual booleans
            bool_columns = ["water", "roads", "buildings"]
            for col in bool_columns:
                df[col] = df[col].astype(bool)

            return df

    except sqlite3.Error as e:
        raise Exception(f"Error reading database: {str(e)}")
