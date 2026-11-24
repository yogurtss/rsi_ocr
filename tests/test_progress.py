import io
import tempfile
import unittest
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rsi_ocr.config import load_config
from rsi_ocr.progress import BatchProgress, overall


@unittest.skipUnless(all(find_spec(name) for name in ("fastapi", "rich", "PIL", "pypdfium2")), "quality extras")
class ProgressTest(unittest.TestCase):
    def test_cached_pages_and_region_models_report_per_document(self):
        from PIL import Image
        from rsi_ocr.capture import capture_plan
        from rsi_ocr.importer import render_pages
        from rsi_ocr.local import run_local
        from rsi_ocr.manager import Manager
        from rsi_ocr.service import Service
        from rsi_ocr.store import write_rows

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "docs"
            source.mkdir()
            with Image.new("RGB", (12, 16), "white") as page:
                page.save(source / "two.pdf", save_all=True, append_images=[page])
                page.save(source / "blank.pdf")
            settings = load_config("configs/import.yaml")
            settings["output_root"] = str(root / "runs")
            batch = root / "runs/batch"
            base = Path("configs/default.yaml")
            config = load_config(base)
            manager = Manager(settings, base, 8520)
            manager.batch_progress = BatchProgress(["oxr", "ovis"], "jev")
            manifest = render_pages(source, batch / "input", settings, progress=lambda _: None)
            plan = capture_plan(config, manifest, batch / "capture")
            for entry in plan:
                page = entry.page
                regions = [{"doc_id": page["doc_id"], "page_id": page["page_id"], "region_id": str(n),
                            "image": str(entry.image), "task": "text", "target": {"status": "ok", "raw": "one",
                            "format": "text"}} for n in range(2) if page["doc_id"] == "two.pdf"]
                write_rows(entry.shard, regions)
            stream = render_pages(source, batch / "input", settings, reuse_existing=True, streaming=True)

            def request(service, path, payload):
                if path.endswith("/score"):
                    return {"results": [{"verdict": "scored", "prob_yes": 0.8} for item in payload["items"]]}
                return {"choices": [{"message": {"content": "one"}, "finish_reason": "stop"}]}

            with patch.object(Service, "request", autospec=True, side_effect=request):
                run_local(config, batch, [(batch / "capture/regions.jsonl", batch / "production")], stream,
                          session=lambda *_: nullcontext(), progress=manager.report_progress,
                          report_work=manager.report_work)
            stages = manager.snapshot()["batch_progress"]["stages"]
            self.assertEqual(stages["render"]["docs"]["two.pdf"], (2, 2, 0))
            self.assertEqual(stages["paddle"]["docs"]["blank.pdf"], (1, 1, 0))
            for name in ("oxr", "ovis", "jev"):
                self.assertEqual(stages[name]["docs"]["two.pdf"], (4, 4, 0))
                self.assertEqual(stages[name]["processed"], 4)
            self.assertEqual(overall(stages), 1)
            self.assertEqual(overall(stages, "blank.pdf"), 1)
            self.assertTrue(any("Jev 4/4 区域" in row["message"] for row in manager.snapshot()["logs"]))

    def test_paged_view_and_log_keep_progress_visible(self):
        from rich.console import Console
        from rsi_ocr.console_view import log_panel, progress_view

        progress = BatchProgress(["oxr", "ovis"], "jev")
        docs = {f"scan-{n:02}.pdf": 3 for n in range(20)}
        for name in progress.stages:
            progress.plan(name, docs, {})
        progress.advance("oxr", "scan-09.pdf", True)
        progress.advance("ovis", "scan-08.pdf", False)
        batch = progress.snapshot()
        progress.advance("oxr", "scan-09.pdf", False)
        self.assertEqual(batch["stages"]["oxr"]["docs"]["scan-09.pdf"], (1, 3, 1))
        console = Console(file=io.StringIO(), width=90, height=20)
        with console.capture() as contents:
            console.print(progress_view(batch, 1, 20, 90))
        output = contents.get()
        self.assertIn("scan-09.pdf", output)
        self.assertNotIn("scan-00.pdf", output)
        self.assertIn("错误 1", output)
        self.assertIn("2/3 页", output)
        with console.capture() as contents:
            console.print(log_panel([{"time": 0, "error": False, "message": "真实日志"}], 5, batch))
        log = contents.get()
        for value in ("总进度", "scan-08.pdf", "OXR 1/60", "Ovis 1/60", "真实日志"):
            self.assertIn(value, log)
