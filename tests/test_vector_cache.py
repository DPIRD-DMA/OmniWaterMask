import sqlite3

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from omniwatermask.vector_cache import (
    DB_NAME,
    GDF_DIR,
    add_to_db,
    check_db,
    initialize_db,
    view_cache_db,
)


class TestInitializeDb:
    def test_creates_database_and_directories(self, cache_dir):
        initialize_db(cache_dir)
        assert (cache_dir / DB_NAME).exists()
        assert (cache_dir / GDF_DIR).is_dir()

    def test_creates_table_schema(self, cache_dir):
        initialize_db(cache_dir)
        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(geodataframes)")
            columns = {row[1] for row in cursor.fetchall()}
            assert columns == {
                "id",
                "polygon",
                "paths",
                "water",
                "roads",
                "buildings",
                "gdf_uid",
            }

    def test_idempotent(self, cache_dir):
        """Calling initialize_db twice should not raise."""
        initialize_db(cache_dir)
        initialize_db(cache_dir)
        assert (cache_dir / DB_NAME).exists()

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        initialize_db(nested)
        assert (nested / DB_NAME).exists()


class TestCheckDb:
    def test_returns_empty_when_no_match(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf, found = check_db(cache_dir, polygon, paths=[])
        assert found is False
        assert len(gdf) == 0

    def test_finds_previously_added_entry(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        paths = []
        test_gdf = gpd.GeoDataFrame(
            geometry=[Point(0.5, 0.5)], crs="EPSG:4326"
        )
        add_to_db(cache_dir, polygon, paths, test_gdf, water=True)

        result_gdf, found = check_db(
            cache_dir, polygon, paths, water=True
        )
        assert found is True
        assert len(result_gdf) == 1

    def test_different_flags_no_match(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        test_gdf = gpd.GeoDataFrame(
            geometry=[Point(0.5, 0.5)], crs="EPSG:4326"
        )
        add_to_db(cache_dir, polygon, [], test_gdf, water=True)

        # Search with roads=True instead — should not match
        _, found = check_db(cache_dir, polygon, [], water=False, roads=True)
        assert found is False


class TestAddToDb:
    def test_adds_entry_and_saves_parquet(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(10, 20, 30, 40)
        gdf = gpd.GeoDataFrame(
            geometry=[box(11, 21, 29, 39)], crs="EPSG:4326"
        )
        add_to_db(cache_dir, polygon, [], gdf, water=True, roads=False, buildings=False)

        # Verify DB row exists
        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM geodataframes")
            assert cursor.fetchone()[0] == 1

        # Verify parquet file exists
        parquet_files = list((cache_dir / GDF_DIR).glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_multiple_entries(self, cache_dir):
        initialize_db(cache_dir)
        for i in range(3):
            polygon = box(i, i, i + 1, i + 1)
            gdf = gpd.GeoDataFrame(
                geometry=[Point(i + 0.5, i + 0.5)], crs="EPSG:4326"
            )
            add_to_db(cache_dir, polygon, [], gdf, water=True)

        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM geodataframes")
            assert cursor.fetchone()[0] == 3


class TestViewCacheDb:
    def test_view_empty_db(self, cache_dir):
        initialize_db(cache_dir)
        df = view_cache_db(cache_dir)
        assert len(df) == 0

    def test_view_populated_db(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, roads=False, buildings=True)

        df = view_cache_db(cache_dir)
        assert len(df) == 1
        assert df["water"].iloc[0]
        assert not df["roads"].iloc[0]
        assert df["buildings"].iloc[0]

    def test_raises_on_missing_db(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            view_cache_db(tmp_path / "nonexistent")
