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
    GENERATION,
    prune_stale_cache,
    view_cache_db,
)

# Entries store geometry buffered in the raster's CRS, so the CRS is part of the
# key. A projected one, since that is what a real caller passes.
CRS_KEY = "EPSG:32750"


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
                "source",
                "ocean",
                "crs",
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
        gdf, found = check_db(cache_dir, polygon, paths=[], crs=CRS_KEY)
        assert found is False
        assert len(gdf) == 0

    def test_finds_previously_added_entry(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        paths = []
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, paths, test_gdf, water=True, crs=CRS_KEY)

        result_gdf, found = check_db(cache_dir, polygon, paths, water=True, crs=CRS_KEY)
        assert found is True
        assert len(result_gdf) == 1

    def test_different_flags_no_match(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], test_gdf, water=True, crs=CRS_KEY)

        # Search with roads=True instead — should not match
        _, found = check_db(
            cache_dir, polygon, [], water=False, roads=True, crs=CRS_KEY
        )
        assert found is False

    def test_different_source_no_match(self, cache_dir):
        """Overture and OSM produce different vectors for the same bounds."""
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(
            cache_dir, polygon, [], test_gdf, water=True, source="overture", crs=CRS_KEY
        )

        _, found = check_db(
            cache_dir, polygon, [], water=True, source="osm", crs=CRS_KEY
        )
        assert found is False

        _, found = check_db(
            cache_dir, polygon, [], water=True, source="overture", crs=CRS_KEY
        )
        assert found is True

    def test_different_ocean_flag_no_match(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], test_gdf, water=True, ocean=True, crs=CRS_KEY)

        _, found = check_db(
            cache_dir, polygon, [], water=True, ocean=False, crs=CRS_KEY
        )
        assert found is False

        _, found = check_db(cache_dir, polygon, [], water=True, ocean=True, crs=CRS_KEY)
        assert found is True


class TestAddToDb:
    def test_adds_entry_and_saves_parquet(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(10, 20, 30, 40)
        gdf = gpd.GeoDataFrame(geometry=[box(11, 21, 29, 39)], crs="EPSG:4326")
        add_to_db(
            cache_dir,
            polygon,
            [],
            gdf,
            water=True,
            roads=False,
            buildings=False,
            crs=CRS_KEY,
        )

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
            gdf = gpd.GeoDataFrame(geometry=[Point(i + 0.5, i + 0.5)], crs="EPSG:4326")
            add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)

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
        add_to_db(
            cache_dir,
            polygon,
            [],
            gdf,
            water=True,
            roads=False,
            buildings=True,
            crs=CRS_KEY,
        )

        df = view_cache_db(cache_dir)
        assert len(df) == 1
        assert df["water"].iloc[0]
        assert not df["roads"].iloc[0]
        assert df["buildings"].iloc[0]

    def test_raises_on_missing_db(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            view_cache_db(tmp_path / "nonexistent")


class TestEmptyBuildIsCached:
    """A build that found nothing must be a hit, not a miss.

    The query that returns nothing is the expensive one - it has to scan the
    whole bbox before it can say so - and it can only ever return nothing again,
    so leaving it uncached makes every later run repeat it.
    """

    def test_none_is_stored_without_a_parquet(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        add_to_db(cache_dir, polygon, [], None, water=True, crs=CRS_KEY)

        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            assert conn.execute("SELECT COUNT(*) FROM geodataframes").fetchone()[0] == 1
        assert list((cache_dir / GDF_DIR).glob("*.parquet")) == []

    def test_empty_frame_is_stored_without_a_parquet(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        add_to_db(cache_dir, polygon, [], gpd.GeoDataFrame(), water=True, crs=CRS_KEY)

        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            assert conn.execute("SELECT COUNT(*) FROM geodataframes").fetchone()[0] == 1
        assert list((cache_dir / GDF_DIR).glob("*.parquet")) == []

    def test_lookup_of_an_empty_build_is_a_hit(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        add_to_db(cache_dir, polygon, [], None, water=True, crs=CRS_KEY)

        gdf, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is True
        assert gdf.empty

    def test_an_unrecorded_query_is_still_a_miss(self, cache_dir):
        """The sentinel must not make every lookup look answered."""
        initialize_db(cache_dir)
        add_to_db(cache_dir, box(0, 0, 1, 1), [], None, water=True, crs=CRS_KEY)

        gdf, found = check_db(cache_dir, box(5, 5, 6, 6), [], water=True, crs=CRS_KEY)
        assert found is False
        assert gdf.empty

    def test_flags_still_discriminate_empty_entries(self, cache_dir):
        """An empty water build says nothing about the roads build."""
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        add_to_db(cache_dir, polygon, [], None, water=True, roads=False, crs=CRS_KEY)

        _, found = check_db(
            cache_dir, polygon, [], water=False, roads=True, crs=CRS_KEY
        )
        assert found is False

    def test_real_geometry_still_round_trips(self, cache_dir):
        """The sentinel path must not disturb the normal one."""
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)

        loaded, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is True
        assert len(loaded) == 1
        assert len(list((cache_dir / GDF_DIR).glob("*.parquet"))) == 1


class TestCacheGeneration:
    def test_db_name_carries_the_generation(self):
        """Entries store already-buffered geometry, so a change to how vectors
        are processed has to be able to invalidate them. The generation lives in
        the filename, which is what makes an older cache invisible rather than
        silently reusable."""
        assert DB_NAME == "geodataframes_v4.db"

    def test_the_parquet_directory_carries_it_too(self):
        """Both halves of a generation move together, or the new one writes
        into the old one's directory and the two become inseparable again."""
        assert GDF_DIR == f"gdfs_v{GENERATION}"
        assert DB_NAME == f"geodataframes_v{GENERATION}.db"

    def test_an_older_generation_is_not_read(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)

        # rename the current db to the previous generation, as an upgrade leaves it
        (cache_dir / DB_NAME).rename(cache_dir / "geodataframes_v3.db")
        initialize_db(cache_dir)

        _, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is False, "a previous generation's entry must not be served"


class TestPruneDropsSupersededGenerations:
    """A generation is a database plus its parquet directory.

    Both carry the version, so the previous generation is identifiable on disk
    rather than having to be deduced from what the current database happens not
    to reference.
    """

    @staticmethod
    def _plant_generation(cache_dir, version, files=2):
        """Write what an install of an earlier version would have left."""
        old_dir = cache_dir / f"gdfs_v{version}"
        old_dir.mkdir(parents=True)
        for i in range(files):
            (old_dir / f"{i}.parquet").write_bytes(b"stale")
        (cache_dir / f"geodataframes_v{version}.db").write_bytes(b"stale")
        return old_dir

    def test_a_previous_generation_is_deleted_whole(self, cache_dir):
        initialize_db(cache_dir)
        old_dir = self._plant_generation(cache_dir, GENERATION - 1)

        assert prune_stale_cache(cache_dir) == 3  # two parquets and the database
        assert not old_dir.exists()
        assert not (cache_dir / f"geodataframes_v{GENERATION - 1}.db").exists()

    def test_the_legacy_unversioned_directory_is_deleted(self, cache_dir):
        """v1 to v3 shared one "gdfs", so it holds every pre-scheme generation
        and is superseded as a whole."""
        initialize_db(cache_dir)
        legacy = cache_dir / "gdfs"
        legacy.mkdir(parents=True)
        (legacy / "old.parquet").write_bytes(b"stale")

        assert prune_stale_cache(cache_dir) == 1
        assert not legacy.exists()

    def test_the_current_generation_survives_and_still_loads(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        self._plant_generation(cache_dir, GENERATION - 1)

        prune_stale_cache(cache_dir)

        loaded, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is True
        assert len(loaded) == 1
        assert (cache_dir / DB_NAME).exists()

    def test_a_newer_generation_is_left_alone(self, cache_dir):
        """It belongs to a newer install sharing this directory, so it is in use
        rather than obsolete. Pruning from the older install must not wipe it."""
        initialize_db(cache_dir)
        newer_dir = cache_dir / f"gdfs_v{GENERATION + 1}"
        newer_dir.mkdir(parents=True)
        (newer_dir / "fresh.parquet").write_bytes(b"fresh")
        newer_db = cache_dir / f"geodataframes_v{GENERATION + 1}.db"
        newer_db.write_bytes(b"fresh")

        assert prune_stale_cache(cache_dir) == 0
        assert (newer_dir / "fresh.parquet").exists()
        assert newer_db.exists()

    def test_missing_cache_dir_is_not_an_error(self, tmp_path):
        assert prune_stale_cache(tmp_path / "nonexistent") == 0

    def test_a_clean_cache_loses_nothing(self, cache_dir):
        initialize_db(cache_dir)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, box(0, 0, 1, 1), [], gdf, water=True, crs=CRS_KEY)

        assert prune_stale_cache(cache_dir) == 0
        assert len(list((cache_dir / GDF_DIR).glob("*.parquet"))) == 1


class TestPruneDropsOrphansInTheCurrentGeneration:
    """Nothing but the current database writes into its own directory, so an
    unreferenced parquet there is an interrupted write, with no ambiguity about
    whether it is really someone else's cache."""

    def test_removes_only_unreferenced_files(self, cache_dir):
        initialize_db(cache_dir)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, box(0, 0, 1, 1), [], gdf, water=True, crs=CRS_KEY)
        orphan = cache_dir / GDF_DIR / "not-referenced-by-any-row.parquet"
        orphan.write_bytes(b"")

        assert prune_stale_cache(cache_dir) == 1
        assert not orphan.exists()
        assert len(list((cache_dir / GDF_DIR).glob("*.parquet"))) == 1

    def test_an_empty_entry_does_not_make_pruning_delete_everything(self, cache_dir):
        """A sentinel row references no file; it must not be read as "this row
        accounts for nothing, so delete the rest"."""
        initialize_db(cache_dir)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, box(0, 0, 1, 1), [], gdf, water=True, crs=CRS_KEY)
        add_to_db(cache_dir, box(5, 5, 6, 6), [], None, water=True, crs=CRS_KEY)

        assert prune_stale_cache(cache_dir) == 0
        assert len(list((cache_dir / GDF_DIR).glob("*.parquet"))) == 1


class TestCrsIsPartOfTheKey:
    """Geometry is buffered in the raster's CRS and stored in it.

    Nothing downstream reprojects, so an entry served to a raster in a different
    CRS rasterizes to a blank or displaced mask rather than raising. The bounding
    box in the key is WGS84 and says nothing about which CRS that was.
    """

    def test_a_different_crs_is_a_miss(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs="EPSG:32750")

        _, found = check_db(cache_dir, polygon, [], water=True, crs="EPSG:32751")
        assert found is False, "an entry buffered in another CRS must not be served"

    def test_the_same_crs_is_a_hit(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs="EPSG:32750")

        _, found = check_db(cache_dir, polygon, [], water=True, crs="EPSG:32750")
        assert found is True

    def test_crs_must_be_given(self, cache_dir):
        """Keyword-only and required, so a caller cannot omit it and silently
        share one key across CRSs."""
        initialize_db(cache_dir)
        with pytest.raises(TypeError):
            check_db(cache_dir, box(0, 0, 1, 1), [], water=True)


class TestStaleEntryIsRepaired:
    """A row whose parquet has gone must not strand its key.

    The row is the only record the file was meant to exist. Reporting a miss and
    leaving it behind makes the caller insert a second row beside it, and the
    next lookup finds the broken one again - so the key refetches forever and
    grows a duplicate row per run.
    """

    def test_missing_file_is_a_miss(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        for parquet in (cache_dir / GDF_DIR).glob("*.parquet"):
            parquet.unlink()

        _, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is False

    def test_missing_file_drops_the_row(self, cache_dir):
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        for parquet in (cache_dir / GDF_DIR).glob("*.parquet"):
            parquet.unlink()

        check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            assert conn.execute("SELECT COUNT(*) FROM geodataframes").fetchone()[0] == 0

    def test_the_key_recovers_on_the_next_build(self, cache_dir):
        """One rebuild after the loss, not one per run forever."""
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        for parquet in (cache_dir / GDF_DIR).glob("*.parquet"):
            parquet.unlink()

        # the run that finds the loss rebuilds and re-adds
        _, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is False
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)

        loaded, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is True
        assert len(loaded) == 1
        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            assert conn.execute("SELECT COUNT(*) FROM geodataframes").fetchone()[0] == 1

    def test_a_sibling_with_its_file_is_still_served(self, cache_dir):
        """Duplicates already on disk from before the repair: the broken row is
        first, but the good one behind it is what the lookup should return."""
        initialize_db(cache_dir)
        polygon = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        stranded = next((cache_dir / GDF_DIR).glob("*.parquet"))
        add_to_db(cache_dir, polygon, [], gdf, water=True, crs=CRS_KEY)
        stranded.unlink()

        loaded, found = check_db(cache_dir, polygon, [], water=True, crs=CRS_KEY)
        assert found is True
        assert len(loaded) == 1
        with sqlite3.connect(cache_dir / DB_NAME) as conn:
            assert conn.execute("SELECT COUNT(*) FROM geodataframes").fetchone()[0] == 1
