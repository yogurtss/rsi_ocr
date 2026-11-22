import time
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from rsi_ocr.config import load_config


@unittest.skipUnless(find_spec("fastapi"), "requires quality extras")
class ManagerTests(unittest.TestCase):
    def test_queue_reuses_healthy_model_and_releases_owned_processes(self):
        from rsi_ocr.manager import Manager

        events = []

        @contextmanager
        def session(config, alias, output, cancelled):
            events.append(("start", alias))
            try:
                yield [Mock(poll=lambda: None)]
            finally:
                events.append(("stop", alias))

        with TemporaryDirectory() as temp, patch("rsi_ocr.manager.model_session", session), patch(
            "rsi_ocr.manager.healthy", return_value=True
        ):
            settings = load_config("configs/import.yaml")
            settings["output_root"] = temp
            settings["deployment"]["models"]["jev"].update(port=19210, render_port=19211)
            manager = Manager(settings, Path("configs/default.yaml"), 19320)
            jev = manager.config["models"]["jev"]
            self.assertEqual(jev["engine_url"], "http://127.0.0.1:19210")
            self.assertEqual(jev["sidecars"]["render"]["argv"][-2:], ["19211", jev["engine_url"]])
            manager.start()
            try:
                manager.enqueue("start", "paddle")
                manager.enqueue("start", "paddle")
                manager.enqueue("start", "oxr")
                last = manager.enqueue("stop", "oxr")
                deadline = time.monotonic() + 3
                while manager.jobs[last]["status"] != "done" and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(manager.jobs[last]["status"], "done")
                self.assertEqual(events, [("start", "paddle"), ("stop", "paddle"), ("start", "oxr"), ("stop", "oxr")])
                self.assertEqual(manager.snapshot()["models"]["oxr"]["status"], "offline")
            finally:
                manager.close()

    def test_managed_models_coexist_and_mode_round_trips(self):
        from rsi_ocr.manager import Manager

        @contextmanager
        def session(*args):
            yield [Mock(poll=lambda: None)]

        with TemporaryDirectory() as directory, patch("rsi_ocr.manager.model_session", session), patch(
            "rsi_ocr.manager.healthy", return_value=True
        ):
            path = Path(directory) / "settings.yaml"
            path.write_text(Path("configs/import.yaml").read_text())
            settings = load_config(path)
            settings["deployment"]["models"]["oxr"].update(gpu="1", gpu_memory=0.42)
            manager = Manager(settings, Path("configs/default.yaml"), 8520, path)
            self.assertEqual(manager.models["oxr"]["gpu"], "1")
            self.assertEqual(manager.models["oxr"]["gpu_memory"], 0.42)
            manager.set_mode("managed")
            self.assertEqual(load_config(path)["deployment"]["mode"], "managed")
            try:
                for alias in ("paddle", "oxr"):
                    manager.load(alias, Path(directory))
                self.assertEqual(set(manager.sessions), {"paddle", "oxr"})
                with self.assertRaises(ValueError):
                    manager.set_mode("on_demand")
                manager.unload("oxr")
                self.assertTrue(manager.is_ready("paddle"))
                manager.set_mode("on_demand")
            finally:
                manager.unload_all()

    def test_managed_inference_overlaps_teachers_and_keeps_services(self):
        from threading import Barrier
        from rsi_ocr.manager import Manager

        stages = []
        teachers = Barrier(2)

        def run_stage(*args, stage, session):
            stages.append(stage)
            if stage in {"oxr", "ovis"}:
                teachers.wait(timeout=2)

        with TemporaryDirectory() as directory:
            settings = load_config("configs/import.yaml")
            settings["output_root"] = directory
            settings["deployment"]["mode"] = "managed"
            manager = Manager(settings, Path("configs/default.yaml"), 8520)
            with patch.object(manager, "is_ready", return_value=True), patch(
                "rsi_ocr.manager.render_pages", return_value=Path("pages.jsonl")
            ), patch("rsi_ocr.manager.resolve_config"), patch("rsi_ocr.manager.run_local", side_effect=run_stage), patch.object(
                manager, "unload_all"
            ) as unload:
                manager.infer({"value": directory})
                self.assertEqual(stages[0], "paddle")
                self.assertEqual(set(stages[1:-1]), {"oxr", "ovis"})
                self.assertEqual(stages[-1], "jev")
                unload.assert_not_called()

    def test_restart_updates_only_selected_model_from_config(self):
        from rsi_ocr.manager import Manager

        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text(Path("configs/import.yaml").read_text())
            settings = load_config(path)
            manager = Manager(settings, Path("configs/default.yaml"), 8520, path)
            path.write_text(path.read_text().replace("port: 8102", "display_name: Ovis Lab, port: 18102"))
            with patch("rsi_ocr.manager.check_checkpoint"), patch.object(manager, "unload") as unload:
                manager.prepare_restart(["ovis"])
            unload.assert_called_once_with("ovis")
            self.assertEqual(manager.models["ovis"]["display_name"], "Ovis Lab")
            self.assertEqual(manager.models["ovis"]["url"], "http://127.0.0.1:18102/v1")
