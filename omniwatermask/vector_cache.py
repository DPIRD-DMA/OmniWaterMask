import json
import logging
import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# The name carries the cache generation, and is bumped rather than migrated so
# an older cache is simply ignored. v2 added the source and ocean columns: v1
# rows carry no record of which vector source produced them. v3 was a content
# change - entries store geometry that has already been buffered, and buffering
# moved from EPSG:3857 to the raster's own projected CRS, which changes the
# distance laid down on the ground by 1/cos(latitude). A v2 row therefore holds
# a buffer this code would no longer produce, and nothing in the row says so.
# v4 adds the crs column, because buffering in the raster's CRS also stores the
# result in it: an entry is only reusable by a raster in the same CRS, and the
# rest of the key does not imply one. Two rasters can share a WGS84 bounding
# box and not a CRS, and rasterize_vector reprojects nothing, so serving the
# wrong one produces a silently blank or displaced mask rather than an error.
#
# Note this is a manual guard: any future change to how vectors are processed
# before they are stored needs another bump, or stale entries are served with
# no indication. Caching the features unbuffered would remove the need.
GENERATION = 4
DB_NAME = f"geodataframes_v{GENERATION}.db"
# The parquet directory carries the generation too, so a bump moves the whole
# cache rather than leaving the new generation writing into the old one's
# directory. That is what made a superseded generation's parquets anonymous:
# once its database was no longer read, nothing on disk said which files had
# belonged to it, and telling them from a genuine orphan needed a scan of every
# row plus a judgement call. A generation is now one database and one directory,
# so dropping it is a delete rather than a deduction. v1-v3 shared an
# unversioned "gdfs", which is therefore superseded as a whole.
GDF_DIR = f"gdfs_v{GENERATION}"

# A build that found no features is recorded with this in place of a uid, and no
# parquet file. Without it, "there is nothing here" is indistinguishable from
# "not looked up yet", so every run repeats a query that can take minutes over a
# remote bbox and can only ever come back empty again.
EMPTY_UID = ""


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
                source TEXT NOT NULL,
                ocean BOOLEAN NOT NULL,
                crs TEXT NOT NULL,
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
    source: str = "overture",
    ocean: bool = False,
    *,
    crs: str,
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
        source: Vector source the entry was built from ("overture" or "osm").
        ocean: Boolean flag for whether ocean water features were included.
        crs: The raster CRS the entry's geometry is stored in, as a string.
            Keyword-only and required: entries are buffered and stored in this
            CRS, and a caller that omitted it would silently match another
            raster's.

    Returns:
        The matching GeoDataFrame if found, otherwise False.
    """
    db_path = cache_dir / DB_NAME
    gdf_dir = cache_dir / GDF_DIR
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, gdf_uid FROM geodataframes
            WHERE polygon = ? AND paths = ?
            AND water = ? AND roads = ? AND buildings = ?
            AND source = ? AND ocean = ? AND crs = ?
        """,
            (
                polygon.wkt,
                json.dumps([str(p) for p in paths]),
                water,
                roads,
                buildings,
                source,
                ocean,
                crs,
            ),
        )
        # All matches, not just the first: a key can hold more than one row, and
        # a row whose parquet has gone is not a reason to ignore a sibling that
        # still has one.
        rows = cursor.fetchall()
        stale_ids = []
        hit = None
        for row_id, gdf_uid in rows:
            if gdf_uid == EMPTY_UID:
                logging.info("Cache records no features for this query")
                hit = (gpd.GeoDataFrame(), True)
                break
            gdf_path = gdf_dir / f"{gdf_uid}.parquet"
            if gdf_path.exists():
                logging.info("Found matching GeoDataFrame in cache")
                logging.info(f"Loading GeoDataFrame from {gdf_path}")
                hit = (gpd.read_parquet(gdf_path), True)  # Load the GDF from disk
                break
            # The row is the only record that this file was ever meant to exist,
            # so leaving it behind strands the key: the miss it forces makes the
            # caller rebuild and insert a second row beside it, and the next
            # lookup finds this one again. Drop it and let the rebuild stand.
            logging.warning(
                f"GeoDataFrame file {gdf_path} not found; dropping the stale entry"
            )
            stale_ids.append(row_id)

        if stale_ids:
            cursor.executemany(
                "DELETE FROM geodataframes WHERE id = ?",
                [(row_id,) for row_id in stale_ids],
            )
            conn.commit()

        if hit is not None:
            return hit
        logging.info("No matching GeoDataFrame found in cache")
        return gpd.GeoDataFrame(), False


def add_to_db(
    cache_dir: Path,
    polygon: Polygon,
    paths: List[Path],
    gdf: Optional[gpd.GeoDataFrame],
    water: bool = False,
    roads: bool = False,
    buildings: bool = False,
    source: str = "overture",
    ocean: bool = False,
    *,
    crs: str,
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
        source: Vector source the entry was built from ("overture" or "osm").
        ocean: Boolean flag for whether ocean water features were included.
        crs: The raster CRS ``gdf`` is stored in, as a string. Keyword-only and
            required, for the reason given on ``check_db``.
        gdf: The GeoDataFrame to store. ``None`` or empty records that the build
            found no features, so the lookup is a hit that returns nothing
            rather than a miss that refetches.
    """
    db_path = cache_dir / DB_NAME
    gdf_dir = cache_dir / GDF_DIR
    if gdf is None or gdf.empty:
        gdf_uid = EMPTY_UID  # nothing to serialise; the row itself is the answer
    else:
        gdf_uid = str(uuid.uuid4())  # Generate a unique identifier
        gdf_file = gdf_dir / f"{gdf_uid}.parquet"  # File name based on UID
        gdf.to_parquet(gdf_file)  # Save the GDF to disk in Parquet format

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO geodataframes
            (polygon, paths, water, roads, buildings, source, ocean, crs, gdf_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                polygon.wkt,
                json.dumps([str(p) for p in paths]),
                water,
                roads,
                buildings,
                source,
                ocean,
                crs,
                gdf_uid,
            ),
        )
        conn.commit()
        logging.info("Added GeoDataFrame to cache")


def _generation_of(name: str) -> Optional[int]:
    """The generation a cache file or directory name carries, if any.

    None means it predates the scheme - the unversioned ``gdfs`` directory that
    v1 to v3 shared - which is superseded by anything with a number.
    """
    match = re.search(r"_v(\d+)(?:\.db)?$", name)
    return int(match.group(1)) if match else None


def prune_stale_cache(cache_dir: Path) -> int:
    """Delete the cache files this version cannot read.

    Two kinds. A superseded generation - a database and the parquet directory
    beside it, both carrying a lower ``GENERATION`` than this one - is deleted
    whole; that is most of what an old cache directory weighs, since a bump
    strands a full copy of the parquets. And within the current generation, a
    parquet no row points at, which an interrupted write leaves behind. Nothing
    else writes into the current generation's directory, so unreferenced there
    means orphaned, with no ambiguity to resolve.

    A *higher* generation is left alone. It belongs to a newer install sharing
    this cache directory, and is in use rather than obsolete.

    This is deliberately not automatic: it destroys the only copy of data an
    older install would still read, so downgrading after a prune means
    refetching. Call it when the reclaimed space is worth that.

    Returns the number of files deleted.
    """
    if not cache_dir.is_dir():
        return 0

    superseded = [
        path
        for path in [*cache_dir.glob("gdfs*"), *cache_dir.glob("geodataframes_*.db")]
        if path.name not in (GDF_DIR, DB_NAME)
        and (_generation_of(path.name) or 0) < GENERATION
    ]

    deleted = 0
    reclaimed = 0
    for path in superseded:
        if path.is_dir():
            files = [child for child in path.rglob("*") if child.is_file()]
            reclaimed += sum(child.stat().st_size for child in files)
            deleted += len(files)
            shutil.rmtree(path)
        elif path.is_file():
            reclaimed += path.stat().st_size
            path.unlink()
            deleted += 1
    if superseded:
        logging.info(
            f"Dropped superseded cache generation(s): "
            f"{', '.join(sorted(path.name for path in superseded))}"
        )

    gdf_dir = cache_dir / GDF_DIR
    db_path = cache_dir / DB_NAME
    if gdf_dir.is_dir():
        referenced: set[str] = set()
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                referenced = {
                    row[0] for row in conn.execute("SELECT gdf_uid FROM geodataframes")
                }
        for parquet in gdf_dir.glob("*.parquet"):
            if parquet.stem not in referenced:
                reclaimed += parquet.stat().st_size
                parquet.unlink()
                deleted += 1

    logging.info(
        f"Pruned {deleted} stale cache file(s) from {cache_dir}, "
        f"reclaiming {reclaimed / 1e6:.1f} MB"
    )
    return deleted


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
            bool_columns = ["water", "roads", "buildings", "ocean"]
            for col in bool_columns:
                df[col] = df[col].astype(bool)

            return df

    except sqlite3.Error as e:
        raise Exception(f"Error reading database: {str(e)}") from e
