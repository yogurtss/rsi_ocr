import json
import time
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from unittest.mock import Mock, patch

from rsi_ocr.config import load_config


@unittest.skipUnless(find_spec("fastapi"), "requires quality extras")
class ManagerTests(unittest.TestCase):
    def wait_for_job(self, manager, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with manager.lock:
                job = manager.jobs[job_id].copy()
            if job["status"] in {"done", "failed"}:
                return job
            time.sleep(.01)
        self.fail(f"Job did not finish: {job}")

    def test_jev_uses_bundled_sdk_and_accepts_external_checkout(self):
        from rsi_ocr.manager import Manager
        from rsi_ocr.service import command

        settings = load_config("configs/import.yaml")
        defaults = Path("configs/default.yaml")
        manager = Manager(settings, defaults, 8520)
        _, _, bundled = command(manager.config, "jev", "render")
        self.assertTrue((bundled / "sdk/app.py").is_file())
        self.assertTrue((bundled / "renderer/assets/katex-0.16.47/katex.min.js").is_file())

        with patch.dict("os.environ", {"JEV_WEIGHTS": "/weights", "MINICPM_OCR_REPO": ""}):
            _, _, legacy = command(load_config(defaults), "jev", "render")
        self.assertEqual(legacy, bundled)

        with TemporaryDirectory() as directory:
            external = Path(directory) / "scripts/minicpm-ocr-jev"
            (external / "sdk").mkdir(parents=True)
            (external / "sdk/app.py").touch()
            settings["deployment"]["minicpm_repo"] = directory
            manager = Manager(settings, defaults, 8520)
            _, _, configured = command(manager.config, "jev", "render")
            self.assertEqual(configured, external)

    def test_queue_reuses_healthy_model_and_releases_owned_processes(self):
        from rsi_ocr.manager import Manager

        events = []

        @contextmanager
        def session(config, alias, output, cancelled):
            events.append(("start", alias))
            if alias == "oxr":
                (output / "oxr.log").write_text(("旧日志\n" * 12000) + "vLLM 启动\n", encoding="utf-8")
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
                self.assertEqual(manager.model_log_tail("oxr"), [])
                manager.enqueue("start", "paddle")
                manager.enqueue("start", "paddle")
                manager.enqueue("start", "oxr")
                last = manager.enqueue("stop", "oxr")
                deadline = time.monotonic() + 3
                while manager.jobs[last]["status"] != "done" and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(manager.jobs[last]["status"], "done")
                self.assertEqual(events, [("start", "paddle"), ("stop", "paddle"), ("start", "oxr"), ("stop", "oxr")])
                snapshot = manager.snapshot()
                self.assertEqual(snapshot["models"]["oxr"]["status"], "offline")
                logs = [entry["message"] for entry in snapshot["logs"]]
                self.assertLess(logs.index("正在加载模型 · paddle · GPU 0"), logs.index("模型已就绪 · paddle"))
                self.assertIn("模型已就绪 · 复用 paddle", logs)
                self.assertIn("模型已卸载 · oxr", logs)
                model_lines = manager.model_log_tail("oxr")
                self.assertEqual(len(model_lines), 64)
                self.assertEqual(model_lines[-1], "vLLM 启动")
                with (Path(temp) / ".manager/oxr.log").open("a", encoding="utf-8") as stream:
                    stream.write("vLLM 新一行\n")
                self.assertEqual(manager.model_log_tail("oxr")[-1], "vLLM 新一行")
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
            settings["output_root"] = directory
            settings["deployment"]["models"]["oxr"].update(gpu="1", gpu_memory=0.42)
            manager = Manager(settings, Path("configs/default.yaml"), 8520, path)
            self.assertEqual(manager.models["oxr"]["gpu"], "1")
            self.assertEqual(manager.models["oxr"]["gpu_memory"], 0.42)
            manager.start()
            try:
                manager.set_mode("managed")
                self.assertEqual(load_config(path)["deployment"]["mode"], "managed")
                self.assertEqual(self.wait_for_job(manager, next(iter(manager.jobs)))["status"], "done")
                self.assertEqual(set(manager.sessions), {"paddle", "oxr", "ovis", "jev"})
                manager.set_mode("on_demand")
                self.assertEqual(self.wait_for_job(manager, list(manager.jobs)[-1])["status"], "done")
                self.assertFalse(manager.sessions)
                self.assertEqual(load_config(path)["deployment"]["mode"], "on_demand")
            finally:
                manager.close()

    def test_managed_startup_is_parallel_and_run_waits_in_queue(self):
        from rsi_ocr.manager import Manager

        launching = Barrier(3)
        release = Event()

        @contextmanager
        def session(config, alias, output, cancelled):
            launching.wait(timeout=3)
            if not release.wait(timeout=3):
                raise TimeoutError("Startup was not released")
            yield [Mock(poll=lambda: None)]

        with TemporaryDirectory() as directory, patch("rsi_ocr.manager.model_session", session), patch(
            "rsi_ocr.manager.healthy", return_value=True
        ), patch("rsi_ocr.manager.render_pages", return_value=Path("pages.jsonl")), patch(
            "rsi_ocr.manager.run_local"
        ) as run:
            root = Path(directory)
            settings = load_config("configs/import.yaml")
            settings["output_root"] = str(root / "output")
            settings["deployment"]["mode"] = "managed"
            for alias in ("ovis", "jev"):
                settings["deployment"]["models"][alias]["enabled"] = False
            source = root / "scan.pdf"
            source.touch()
            manager = Manager(settings, Path("configs/default.yaml"), 8520)
            manager.start()
            try:
                launching.wait(timeout=3)
                models = manager.snapshot()["models"]
                self.assertEqual({alias for alias, model in models.items() if model["status"] == "starting"},
                                 {"paddle", "oxr"})
                job_id = manager.enqueue("run", str(source))
                self.assertEqual(manager.jobs[job_id]["status"], "queued")
                run.assert_not_called()
                release.set()
                self.assertEqual(self.wait_for_job(manager, job_id)["status"], "done")
                self.assertEqual([call.kwargs["stage"] for call in run.call_args_list], ["paddle", "oxr", "mine"])
                self.assertEqual(set(manager.sessions), {"paddle", "oxr"})
            finally:
                release.set()
                manager.close()

    def test_managed_inference_overlaps_teachers_and_keeps_services(self):
        from rsi_ocr.manager import Manager

        stages = []
        teachers = Barrier(2)

        def run_stage(*args, stage, session, progress, check_cancelled, layout_session, report_page):
            stages.append(stage)
            progress(f"执行 {stage}")
            if stage == "paddle":
                with layout_session() as report_device:
                    report_device("gpu:2")
                    self.assertEqual(manager.snapshot()["layout_runtime"], {"status": "running", "device": "gpu:2"})
            if stage in {"oxr", "ovis"}:
                teachers.wait(timeout=2)

        with TemporaryDirectory() as directory:
            settings = load_config("configs/import.yaml")
            settings["output_root"] = directory
            settings["deployment"]["mode"] = "managed"
            source = Path(directory) / "scan.pdf"
            source.touch()
            manager = Manager(settings, Path("configs/default.yaml"), 8520)
            with patch.object(manager, "is_ready", return_value=True), patch(
                "rsi_ocr.manager.render_pages", return_value=Path("pages.jsonl")
            ), patch("rsi_ocr.manager.resolve_config"), patch("rsi_ocr.manager.run_local", side_effect=run_stage), patch.object(
                manager, "unload_all"
            ) as unload:
                manager.infer({"value": str(source)})
                self.assertEqual(stages[0], "paddle")
                self.assertEqual(set(stages[1:-1]), {"oxr", "ovis"})
                self.assertEqual(stages[-1], "jev")
                self.assertEqual(manager.snapshot()["layout_runtime"], {"status": "done", "device": "gpu:2"})
                self.assertIn("执行 jev", [entry["message"] for entry in manager.snapshot()["logs"]])
                unload.assert_not_called()

    def test_shutdown_unloads_model_before_waiting_for_active_job(self):
        from rsi_ocr.manager import Manager

        active = Event()
        released = Event()

        @contextmanager
        def session(*args):
            try:
                yield [Mock(poll=lambda: None)]
            finally:
                released.set()

        def infer(job):
            active.set()
            if not released.wait(timeout=2):
                raise RuntimeError("推理仍在等待模型")
            manager.check_closing()

        with TemporaryDirectory() as directory, patch("rsi_ocr.manager.model_session", session):
            settings = load_config("configs/import.yaml")
            settings["output_root"] = str(Path(directory) / "output")
            manager = Manager(settings, Path("configs/default.yaml"), 8520)
            with patch.object(manager, "infer", side_effect=infer):
                manager.start()
                try:
                    start_id = manager.enqueue("start", "paddle")
                    self.assertEqual(self.wait_for_job(manager, start_id)["status"], "done")
                    active_id = manager.enqueue("run", directory)
                    self.assertTrue(active.wait(timeout=2))
                    queued_id = manager.enqueue("run", directory)
                finally:
                    manager.close()

            snapshot = manager.snapshot()
            jobs = {job["id"]: job for job in snapshot["jobs"]}
            self.assertEqual(jobs[active_id]["detail"], "后台正在关闭")
            self.assertEqual(jobs[queued_id]["detail"], "后台已关闭")
            self.assertEqual(snapshot["models"]["paddle"]["status"], "offline")
            self.assertTrue(manager.closed.is_set())

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

    def test_concurrency_change_waits_for_current_job_and_restarts_selected_model(self):
        from rsi_ocr.manager import Manager

        active, release = Event(), Event()
        launches = []

        @contextmanager
        def session(config, alias, output, cancelled):
            spec = config["models"][alias]
            argv = spec["launch"]["argv"]
            launches.append((alias, spec["concurrency"], argv[argv.index("--max-num-seqs") + 1]))
            yield [Mock(poll=lambda: None)]

        def infer(job):
            active.set()
            if not release.wait(timeout=3):
                raise TimeoutError("Test batch did not finish")

        with TemporaryDirectory() as directory, patch("rsi_ocr.manager.model_session", session):
            root = Path(directory)
            path = root / "settings.yaml"
            path.write_text(Path("configs/import.yaml").read_text())
            settings = load_config(path)
            settings["output_root"] = str(root / "out")
            base = Path("configs/default.yaml")
            manager = Manager(settings, base, 8520, path)
            with patch.object(manager, "infer", side_effect=infer):
                manager.start()
                try:
                    start = manager.enqueue("start", "paddle")
                    self.assertEqual(self.wait_for_job(manager, start)["status"], "done")
                    manager.enqueue("run", directory)
                    self.assertTrue(active.wait(timeout=2))
                    with self.assertRaisesRegex(ValueError, "正整数"):
                        manager.enqueue("concurrency", "paddle", 0)
                    change = manager.enqueue("concurrency", "paddle", 8)
                    self.assertEqual(manager.jobs[change]["status"], "queued")
                    self.assertEqual(manager.snapshot()["models"]["paddle"]["max_sequences"], 6)
                    release.set()
                    self.assertEqual(self.wait_for_job(manager, change)["status"], "done")
                    self.assertEqual(manager.snapshot()["models"]["paddle"]["max_sequences"], 8)
                    self.assertEqual(manager.snapshot()["models"]["oxr"]["max_sequences"], 6)
                finally:
                    release.set()
                    manager.close()
            self.assertEqual(launches, [("paddle", 6, "6"), ("paddle", 8, "8")])
            self.assertEqual(Manager(load_config(path), base, 8520).models["paddle"]["max_sequences"], 8)

    def test_model_settings_round_trip_and_unload_changed_service(self):
        from rsi_ocr.manager import Manager, ModelSettings

        @contextmanager
        def session(*args):
            yield [Mock(poll=lambda: None)]

        with TemporaryDirectory() as directory, patch("rsi_ocr.manager.model_session", session):
            path = Path(directory) / "settings.yaml"
            path.write_text(Path("configs/import.yaml").read_text())
            base = Path("configs/default.yaml")
            manager = Manager(load_config(path), base, 8520, path)
            manager.set_mode("managed")
            manager.load("jev", Path(directory))
            manager.load("paddle", Path(directory))
            weights = Path(directory) / "new-ovis"
            manager.set_deployment("managed", {
                "jev": ModelSettings(enabled=False, path="MiniCPM-OCR-Jev", gpu=""),
                "ovis": ModelSettings(enabled=True, path=str(weights), gpu="2"),
            })
            settings = load_config(path)
            reloaded = Manager(settings, base, 8520, path)
            self.assertEqual(set(manager.sessions), {"paddle"})
            self.assertEqual(manager.models["jev"]["status"], "disabled")
            self.assertIsNone(reloaded.config["mining"]["judge"])
            self.assertEqual(reloaded.models["ovis"]["path"], str(weights))
            self.assertEqual(reloaded.config["models"]["ovis"]["launch"]["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(settings["deployment"]["models"]["jev"]["render_port"], 18500)
            with self.assertRaisesRegex(ValueError, "已在模型设置中关闭"):
                manager.enqueue("start", "jev")
            saved = path.read_text()
            with self.assertRaisesRegex(ValueError, "必须启用"):
                manager.set_deployment("managed", {
                    "paddle": ModelSettings(enabled=False, path="paddleocr-vl-16", gpu="0"),
                })
            self.assertEqual(path.read_text(), saved)
            self.assertTrue(manager.models["paddle"]["enabled"])
            manager.unload_all()

    def test_layout_model_settings_migrate_legacy_keys_and_preserve_model_devices(self):
        from rsi_ocr.manager import DeploymentSettings, LayoutSettings, Manager

        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            base = Path("configs/default.yaml")
            settings = load_config("configs/import.yaml")
            settings["deployment"]["models"].pop("layout")
            settings["deployment"].update(layout_path="old-layout", layout_device="gpu:1")
            path.write_text(json.dumps(settings))
            manager = Manager(settings, base, 8520, path)
            self.assertEqual(manager.snapshot()["layout"], {"path": "old-layout", "device": "gpu:1"})

            change = DeploymentSettings(
                mode="on_demand", models={"layout": {"path": "new-layout", "device": "gpu:2"}}
            )
            manager.set_deployment(change.mode, change.models)
            manager.set_deployment("on_demand", {})
            saved = load_config(path)
            self.assertNotIn("layout_path", saved["deployment"])
            self.assertNotIn("layout_device", saved["deployment"])
            self.assertEqual(saved["deployment"]["models"]["layout"], {"path": "new-layout", "device": "gpu:2"})
            reloaded = Manager(saved, base, 8520, path)
            self.assertEqual(reloaded.snapshot()["layout"], manager.snapshot()["layout"])
            pipeline = reloaded.config["capture"]["pipeline"]
            self.assertEqual(pipeline["device"], "gpu:2")
            self.assertEqual(Path(pipeline["layout_detection_model_dir"]).name, "new-layout")
            self.assertEqual(reloaded.models["paddle"]["gpu"], "0")

            with self.assertRaisesRegex(ValueError, r"deployment\.models\.layout\.device"):
                manager.set_deployment("on_demand", {"layout": LayoutSettings(path="new-layout", device="gpu:-1")})
            self.assertEqual(load_config(path), saved)
            manager.set_deployment("on_demand", {}, layout_device="cpu")
            self.assertEqual(manager.snapshot()["layout"], {"path": "new-layout", "device": "cpu"})

    def test_managed_inference_with_only_paddle(self):
        from rsi_ocr.manager import Manager

        with TemporaryDirectory() as directory:
            settings = load_config("configs/import.yaml")
            settings["output_root"] = directory
            settings["deployment"]["mode"] = "managed"
            for alias in ("oxr", "ovis", "jev"):
                settings["deployment"]["models"][alias]["enabled"] = False
            source = Path(directory) / "scan.pdf"
            source.touch()
            manager = Manager(settings, Path("configs/default.yaml"), 8520)
            with patch.object(manager, "is_ready", return_value=True) as ready, patch(
                "rsi_ocr.manager.render_pages", return_value=Path("pages.jsonl")
            ), patch("rsi_ocr.manager.run_local") as run:
                manager.infer({"value": str(source)})
            ready.assert_called_once_with("paddle")
            self.assertEqual([call.kwargs["stage"] for call in run.call_args_list], ["paddle", "mine"])
