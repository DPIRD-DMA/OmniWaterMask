"""Tests for Overture release resolution and its S3 fallback."""

from unittest.mock import patch

import pyarrow.fs as fs
import pytest

from omniwatermask import overture_source as osrc


class _FakeInfo:
    def __init__(self, base_name, type_=fs.FileType.Directory):
        self.base_name = base_name
        self.type = type_


class _FakeS3:
    """Minimal stand-in for S3FileSystem: prefix -> child names."""

    def __init__(self, tree):
        self.tree = tree

    def get_file_info(self, selector):
        return [_FakeInfo(name) for name in self.tree.get(selector.base_dir, [])]


@pytest.fixture(autouse=True)
def _clear_release_cache():
    osrc._resolved_release = None
    yield
    osrc._resolved_release = None


def _tree(**releases):
    tree = {osrc.RELEASE_PREFIX: list(releases)}
    for release, themes in releases.items():
        tree[f"{osrc.RELEASE_PREFIX}/{release}"] = themes
    return tree


ALL_THEMES = sorted(osrc.REQUIRED_THEMES) + ["theme=places"]


def test_uses_overture_discovery_when_available():
    with patch.object(osrc.core, "get_latest_release", return_value="2026-07-22.0"):
        assert osrc._resolve_release() == "2026-07-22.0"


def test_falls_back_to_s3_when_discovery_fails():
    tree = _tree(**{"2026-06-17.0": ALL_THEMES, "2026-07-22.0": ALL_THEMES})
    with (
        patch.object(osrc.core, "get_latest_release", side_effect=Exception("404")),
        patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(tree)),
    ):
        assert osrc._resolve_release() == "2026-07-22.0"


def test_skips_incomplete_release_for_older_complete_one():
    tree = _tree(**{"2026-06-17.0": ALL_THEMES, "2026-07-22.0": ["theme=places"]})
    with (
        patch.object(osrc.core, "get_latest_release", side_effect=Exception("404")),
        patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(tree)),
    ):
        assert osrc._resolve_release() == "2026-06-17.0"


def test_sorts_release_sequence_numerically():
    tree = _tree(**{"2026-07-22.9": ALL_THEMES, "2026-07-22.10": ALL_THEMES})
    with patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(tree)):
        assert osrc._latest_release_from_s3() == "2026-07-22.10"


def test_incomplete_upstream_type_map_uses_known_themes(caplog):
    with caplog.at_level("WARNING"):
        themes = osrc._required_themes({"water": "base"})

    assert themes == osrc.KNOWN_REQUIRED_THEMES
    assert "using the known theme set" in caplog.text


def test_raises_with_osm_hint_when_both_paths_fail():
    with (
        patch.object(osrc.core, "get_latest_release", side_effect=Exception("404")),
        patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(_tree())),
    ):
        with pytest.raises(RuntimeError, match='vector_source="osm"'):
            osrc._resolve_release()


def test_release_is_resolved_once_per_process():
    with patch.object(
        osrc.core, "get_latest_release", return_value="2026-07-22.0"
    ) as discovery:
        osrc._resolve_release()
        osrc._resolve_release()
    assert discovery.call_count == 1


def test_discovery_failure_does_not_enter_the_retry_backoff():
    with (
        patch.object(osrc.core, "get_latest_release", side_effect=Exception("404")),
        patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(_tree())),
        patch.object(osrc.time, "sleep") as sleep,
    ):
        with pytest.raises(RuntimeError):
            osrc._fetch_with_retry("water", (0.0, 0.0, 1.0, 1.0), kind="water")
    sleep.assert_not_called()


def test_degraded_catalogue_without_a_latest_release_falls_back():
    """A reachable catalogue whose "latest" is null returns None rather than
    raising. It must not become the literal release id "None"."""
    tree = _tree(**{"2026-07-22.0": ALL_THEMES})
    with (
        patch.object(osrc.core, "get_latest_release", return_value=None),
        patch.object(osrc.fs, "S3FileSystem", return_value=_FakeS3(tree)),
    ):
        assert osrc._resolve_release() == "2026-07-22.0"


def test_release_is_forwarded_to_the_reader():
    with (
        patch.object(osrc.core, "get_latest_release", return_value="2026-07-22.0"),
        patch.object(osrc.core, "record_batch_reader", return_value=None) as reader,
    ):
        osrc._fetch_once("water", (0.0, 0.0, 1.0, 1.0), "2026-07-22.0")
    assert reader.call_args.kwargs["release"] == "2026-07-22.0"
    assert reader.call_args.kwargs["stac"] is True


def test_release_is_forwarded_to_the_empty_bbox_probe():
    with patch.object(osrc.core, "_prepare_query", return_value=None) as prepare:
        assert osrc._bbox_has_no_files("water", (0.0, 0.0, 1.0, 1.0), "2026-07-22.0")
    assert prepare.call_args.kwargs["release"] == "2026-07-22.0"


def test_invalidation_leaves_a_newer_release_alone():
    """A slow fetch failing against a pruned release must not discard a newer
    one another thread has since resolved."""
    osrc._resolved_release = "2026-07-22.0"
    osrc._invalidate_release("2026-06-17.0")
    assert osrc._resolved_release == "2026-07-22.0"

    osrc._invalidate_release("2026-07-22.0")
    assert osrc._resolved_release is None


def test_exhausted_fetch_does_not_discard_a_concurrently_resolved_release():
    """The losing thread's failure must not erase the release the winner cached."""

    def _fail_and_let_another_thread_win(*args, **kwargs):
        # Stand in for a second thread rediscovering while this fetch is in
        # flight against the older release.
        osrc._resolved_release = "2026-09-15.0"
        raise Exception("gone")

    with (
        patch.object(osrc.core, "get_latest_release", return_value="2026-07-22.0"),
        patch.object(osrc, "_fetch_once", side_effect=_fail_and_let_another_thread_win),
        patch.object(osrc.time, "sleep"),
    ):
        with pytest.raises(RuntimeError):
            osrc._fetch_with_retry("water", (0.0, 0.0, 1.0, 1.0), kind="water")

    assert osrc._resolved_release == "2026-09-15.0"


def test_exhausted_fetch_invalidates_the_cached_release():
    """Overture prunes releases monthly, so a long-lived process must not keep
    asking for one that has since been deleted."""
    with (
        patch.object(osrc.core, "get_latest_release", return_value="2026-07-22.0"),
        patch.object(osrc, "_fetch_once", side_effect=Exception("gone")),
        patch.object(osrc.time, "sleep"),
    ):
        with pytest.raises(RuntimeError):
            osrc._fetch_with_retry("water", (0.0, 0.0, 1.0, 1.0), kind="water")
    assert osrc._resolved_release is None
