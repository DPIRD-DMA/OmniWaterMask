import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import rasterio.shutil
import torch

from omniwatermask.target_builders import TargetBuildError
from omniwatermask.water_inf_pipeline import collect_models, make_water_mask_debug


class TestCollectModels:
    @patch("omniwatermask.water_inf_pipeline.load_model")
    def test_loads_single_custom_model(self, mock_load):
        mock_load.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path="/fake/model.pth",
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 1
        mock_load.assert_called_once()

    @patch("omniwatermask.water_inf_pipeline.load_model")
    def test_loads_multiple_custom_models(self, mock_load):
        mock_load.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path=["/fake/m1.pth", "/fake/m2.pth"],
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 2
        assert mock_load.call_count == 2

    @patch("omniwatermask.water_inf_pipeline.load_model_from_weights")
    @patch("omniwatermask.water_inf_pipeline.get_models")
    def test_loads_default_models(self, mock_get_models, mock_load_weights):
        mock_get_models.return_value = [
            {
                "Path": Path("/fake/model.pth"),
                "timm_model_name": "convnextv2_base",
                "model_library": "fastai",
            }
        ]
        mock_load_weights.return_value = MagicMock(spec=torch.nn.Module)
        models = collect_models(
            model_path="",
            destination_model_dir=None,
            model_download_source="hugging_face",
            inference_device=torch.device("cpu"),
            inference_dtype=torch.float32,
        )
        assert len(models) == 1
        mock_load_weights.assert_called_once()


class TestMakeWaterMaskDebug:
    def test_raises_without_model_or_ndwi_and_no_vectors(self, sample_geotiff):
        with pytest.raises(ValueError, match="must enable use_model"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=False,
                use_ndwi=True,
            )

    def test_raises_without_ndwi_and_no_vectors(self, sample_geotiff):
        with pytest.raises(ValueError, match="must enable use_ndwi"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=True,
                use_ndwi=False,
            )

    def test_accepts_string_scene_path(self, sample_geotiff):
        """Verify string paths are accepted (validation only, not full run)."""
        with pytest.raises(ValueError, match="must enable use_model"):
            make_water_mask_debug(
                scene_paths=str(sample_geotiff),
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=False,
                use_ndwi=True,
            )

    def test_rejects_invalid_scene_paths_type(self):
        with pytest.raises(ValueError, match="scene_paths must be"):
            make_water_mask_debug(
                scene_paths=123,
                band_order=[1, 2, 3, 4],
            )

    def test_rejects_invalid_vector_source(self, sample_geotiff):
        """Catching this up front fails the call before any scene is opened,
        rather than after a per-scene build has already failed in its thread."""
        with pytest.raises(ValueError, match="Unknown vector_source"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                vector_source="openstreetmap",
            )

    def test_rejects_an_outdated_osmnx_up_front(self, sample_geotiff, monkeypatch):
        """build_targets checks this too, but only inside its thread - where the
        raise is lost and the caller sees nothing but a thread that exited
        without a result. Checking here surfaces the message that says what to
        do about it."""
        outdated = type("FakeOx", (), {"__version__": "1.9.0"})
        monkeypatch.setattr("omniwatermask.target_builders.ox", outdated)

        with pytest.raises(ImportError, match="too old"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                vector_source="osm",
            )

    def test_outdated_osmnx_does_not_block_overture(self, sample_geotiff, monkeypatch):
        """osmnx is only used for the Overpass path, so its version must not
        gate a run that never touches it."""
        outdated = type("FakeOx", (), {"__version__": "1.9.0"})
        monkeypatch.setattr("omniwatermask.target_builders.ox", outdated)

        # Fails on an unrelated argument check, i.e. it got past the ox version.
        with pytest.raises(ValueError, match="must enable use_model"):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                use_osm_water=False,
                use_model=False,
                use_ndwi=True,
                vector_source="overture",
            )

    def test_accepts_both_valid_vector_sources(self, sample_geotiff):
        """Guards against the validation list drifting from VECTOR_SOURCES."""
        for source in ("overture", "osm"):
            with pytest.raises(ValueError, match="must enable use_model"):
                make_water_mask_debug(
                    scene_paths=[sample_geotiff],
                    band_order=[1, 2, 3, 4],
                    use_osm_water=False,
                    use_model=False,
                    use_ndwi=True,
                    vector_source=source,
                )


@pytest.fixture
def stub_pipeline(tmp_path):
    """Stub out model loading and detection so only argument forwarding runs."""
    import numpy as np

    with (
        patch("omniwatermask.water_inf_pipeline.collect_models", return_value=[]),
        patch(
            "omniwatermask.water_inf_pipeline.integrate_water_detection_methods"
        ) as mock_integrate,
    ):
        mock_integrate.return_value = (
            np.zeros((1, 100, 100), dtype=np.uint8),
            ["water_mask"],
            None,
        )
        yield mock_integrate


class TestVectorSourceForwarding:
    """vector_source and include_ocean must reach the detection layer.

    They are threaded through four call sites; a dropped kwarg would silently
    fall back to the default source rather than fail.
    """

    def test_debug_forwards_defaults(self, sample_geotiff, stub_pipeline, tmp_path):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
        )
        kwargs = stub_pipeline.call_args.kwargs
        assert kwargs["vector_source"] == "overture"
        assert kwargs["include_ocean"] is True

    def test_debug_forwards_explicit_values(
        self, sample_geotiff, stub_pipeline, tmp_path
    ):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            vector_source="osm",
            include_ocean=False,
        )
        kwargs = stub_pipeline.call_args.kwargs
        assert kwargs["vector_source"] == "osm"
        assert kwargs["include_ocean"] is False

    def test_make_water_mask_forwards_to_debug(
        self, sample_geotiff, stub_pipeline, tmp_path
    ):
        from omniwatermask import make_water_mask

        make_water_mask(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            vector_source="osm",
            include_ocean=False,
        )
        kwargs = stub_pipeline.call_args.kwargs
        assert kwargs["vector_source"] == "osm"
        assert kwargs["include_ocean"] is False


class TestSkipsSceneWhenTargetsFail:
    """A mask built without its vector targets looks plausible on inspection.

    Exporting one would put a quietly wrong file on disk, and its presence would
    make the next run skip the scene. Writing nothing keeps the batch going and
    leaves the scene to be retried.
    """

    @pytest.fixture
    def out_dir(self, tmp_path):
        """Keep outputs out of tmp_path, where the input scene already sits."""
        return tmp_path / "output"

    def test_writes_no_output(self, sample_geotiff, stub_pipeline, out_dir):
        stub_pipeline.side_effect = TargetBuildError("Could not build vector targets")
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=out_dir,
        )
        assert list(out_dir.glob("*.tif")) == []

    def test_omits_the_scene_from_the_returned_paths(
        self, sample_geotiff, stub_pipeline, out_dir
    ):
        """A returned path that was never written would break any caller that
        goes on to open it."""
        stub_pipeline.side_effect = TargetBuildError("Could not build vector targets")
        paths = make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=out_dir,
        )
        assert paths == []

    def test_keeps_processing_the_rest_of_the_batch(
        self, sample_geotiff, stub_pipeline, tmp_path, out_dir
    ):
        """One transient outage must not cost the remaining scenes."""
        import shutil

        import numpy as np

        second_scene = tmp_path / "second_scene.tif"
        shutil.copy(sample_geotiff, second_scene)

        stub_pipeline.side_effect = [
            TargetBuildError("Could not build vector targets"),
            (np.zeros((1, 100, 100), dtype=np.uint8), ["water_mask"], None),
        ]
        paths = make_water_mask_debug(
            scene_paths=[sample_geotiff, second_scene],
            band_order=[1, 2, 3, 4],
            output_dir=out_dir,
        )
        assert len(paths) == 1
        assert paths[0].stem.startswith("second_scene")
        assert paths[0].exists()

    def test_a_rerun_retries_the_skipped_scene(
        self, sample_geotiff, stub_pipeline, out_dir
    ):
        """Leaving no file behind is what makes the retry happen: the existing
        output check is the only thing that would skip it."""
        import numpy as np

        stub_pipeline.side_effect = TargetBuildError("Could not build vector targets")
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=out_dir,
        )

        stub_pipeline.side_effect = None
        stub_pipeline.return_value = (
            np.zeros((1, 100, 100), dtype=np.uint8),
            ["water_mask"],
            None,
        )
        paths = make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=out_dir,
            overwrite=False,
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_unrelated_failures_still_propagate(
        self, sample_geotiff, stub_pipeline, out_dir
    ):
        """Only a target build failure is a skip; anything else is a real bug."""
        stub_pipeline.side_effect = MemoryError("out of memory")
        with pytest.raises(MemoryError):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                output_dir=out_dir,
            )

    def test_a_fetch_error_travels_the_whole_real_path(
        self, sample_geotiff, out_dir, caplog
    ):
        """End to end without stubbing the detection layer: the error has to
        cross a thread boundary and a queue to reach the skip, and every link is
        somewhere a failure could be dropped instead."""
        with (
            patch("omniwatermask.water_inf_pipeline.collect_models", return_value=[]),
            patch(
                "omniwatermask.target_builders.get_overture_features",
                side_effect=OSError("connection reset"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            paths = make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                output_dir=out_dir,
                use_model=False,
                use_ndwi=True,
                use_osm_water=True,
            )

        assert paths == []
        assert list(out_dir.glob("*.tif")) == []
        assert "could not be built" in caplog.text

    def test_a_negative_target_failure_also_skips(self, sample_geotiff, out_dir):
        """Water and the building/road masks are built in separate threads. A
        mask missing only its negative targets over-detects on roofs and roads,
        which is just as wrong as one missing its water targets."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        def fail_only_negative_kinds(*args, **kwargs):
            if kwargs.get("kind") == "water":
                return gpd.GeoDataFrame(
                    geometry=[shapely_box(390200, 6460200, 390800, 6460800)],
                    crs="EPSG:32650",
                )
            raise OSError("connection reset")

        with (
            patch("omniwatermask.water_inf_pipeline.collect_models", return_value=[]),
            patch(
                "omniwatermask.target_builders.get_overture_features",
                side_effect=fail_only_negative_kinds,
            ),
        ):
            paths = make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                output_dir=out_dir,
                use_model=False,
                use_ndwi=True,
                use_osm_water=True,
                use_osm_building=True,
                use_osm_roads=True,
            )

        assert paths == []
        assert list(out_dir.glob("*.tif")) == []

    def test_leaves_no_thread_running_after_a_skip(self, sample_geotiff, out_dir):
        """Positive and negative targets are built in two threads. Propagating
        the first failure without joining the second would leave it fetching
        into the next scene, and an outage would pile those up across a batch."""
        import threading
        import time

        def fail_fast_on_water_slowly_otherwise(*args, **kwargs):
            # The negative targets must still be in flight when the positive
            # build fails, or an unjoined thread would finish before the check.
            if kwargs.get("kind") != "water":
                time.sleep(0.5)
            raise OSError("connection reset")

        def live_worker_threads():
            # tqdm runs a daemon monitor thread of its own; the target threads
            # are non-daemon, so restrict the comparison to those.
            return {t for t in threading.enumerate() if not t.daemon}

        before = live_worker_threads()
        with (
            patch("omniwatermask.water_inf_pipeline.collect_models", return_value=[]),
            patch(
                "omniwatermask.target_builders.get_overture_features",
                side_effect=fail_fast_on_water_slowly_otherwise,
            ),
        ):
            make_water_mask_debug(
                scene_paths=[sample_geotiff],
                band_order=[1, 2, 3, 4],
                output_dir=out_dir,
                use_model=False,
                use_ndwi=True,
                use_osm_water=True,
                use_osm_building=True,
                use_osm_roads=True,
            )

        assert live_worker_threads() - before == set()


class TestResampleIsVirtual:
    """The resampled scene is a /vsimem VRT, never a raster on disk.

    The pipeline used to hand the resample around as a written GeoTIFF, so the
    scene path, the output name and the bands read all came from that file.
    They now come from three separate places, and nothing on disk would catch
    it if one of them drifted.
    """

    def _integrate_call(self, stub_pipeline):
        assert stub_pipeline.call_args is not None
        return stub_pipeline.call_args.kwargs

    def test_reads_bands_at_the_resampled_size(
        self, sample_geotiff, stub_pipeline, tmp_path
    ):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        # The 100x100 10m fixture at 20m is 50x50.
        assert self._integrate_call(stub_pipeline)["input_bands"].shape == (4, 50, 50)

    def test_passes_the_vrt_as_the_scene(self, sample_geotiff, stub_pipeline, tmp_path):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        assert self._integrate_call(stub_pipeline)["input_path"].startswith("/vsimem/")

    def test_output_keeps_the_resample_suffix(
        self, sample_geotiff, stub_pipeline, tmp_path
    ):
        """The suffix used to come from the written file's stem."""
        outputs = make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        assert outputs[0].name.startswith(f"{sample_geotiff.stem}_resample_20m_")
        assert outputs[0].exists()

    def test_writes_no_resampled_raster(self, sample_geotiff, stub_pipeline, tmp_path):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        written = {p.name for p in tmp_path.iterdir()} | {
            p.name for p in sample_geotiff.parent.iterdir()
        }
        assert not [n for n in written if n.endswith("_resample_20m.tif")]

    def test_releases_the_vrt(self, sample_geotiff, stub_pipeline, tmp_path):
        """/vsimem is process-global; a batch that never freed it would grow."""
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        vrt_path = self._integrate_call(stub_pipeline)["input_path"]
        assert not rasterio.shutil.exists(vrt_path)

    def test_releases_the_vrt_when_the_scene_fails(
        self, sample_geotiff, stub_pipeline, tmp_path
    ):
        """A scene skipped mid-run must not leak its VRT to the end of the batch."""
        stub_pipeline.side_effect = TargetBuildError("no vectors")
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
            resample_res=20,
        )
        vrt_path = self._integrate_call(stub_pipeline)["input_path"]
        assert not rasterio.shutil.exists(vrt_path)

    def test_no_vrt_without_resampling(self, sample_geotiff, stub_pipeline, tmp_path):
        make_water_mask_debug(
            scene_paths=[sample_geotiff],
            band_order=[1, 2, 3, 4],
            output_dir=tmp_path,
        )
        kwargs = self._integrate_call(stub_pipeline)
        assert kwargs["input_path"] == sample_geotiff
        assert kwargs["input_bands"].shape == (4, 100, 100)
