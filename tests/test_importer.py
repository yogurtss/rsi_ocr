import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rsi_ocr.config import load_config
from rsi_ocr.importer import render_pages
from rsi_ocr.store import read_rows


@unittest.skipUnless(find_spec("pypdfium2") and find_spec("PIL"), "requires quality extras")
class InputFolderTest(unittest.TestCase):
    def test_pdf_pages_images_and_nested_result_exclusion(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "batch" / "input"
            nested = root / "nested"
            nested.mkdir()
            picture = nested / "sample.JPEG"
            with Image.new("RGB", (24, 36), "white") as page:
                page.save(picture)
                page.save(root / "sample.PDF", save_all=True, append_images=[page], resolution=72)
            settings = {"extensions": [".pdf", ".jpeg"], "recursive": True, "dpi": 144,
                        "output_root": root / "results"}
            progress = []
            rows = list(read_rows(render_pages(root, output, settings, progress=progress.append)))
            self.assertEqual(progress, [
                "导入文件 1/2 · nested/sample.JPEG",
                "导入文件 2/2 · sample.PDF",
                "PDF · 第 1/2 页 · 文件 2/2 · sample.PDF",
                "PDF · 第 2/2 页 · 文件 2/2 · sample.PDF",
            ])
            self.assertEqual([(r["doc_id"], r["page_id"]) for r in rows],
                             [("nested/sample.JPEG", 1), ("sample.PDF", 1), ("sample.PDF", 2)])
            self.assertEqual((output / rows[0]["image"]).read_bytes(), picture.read_bytes())
            with Image.open(output / rows[1]["image"]) as page:
                self.assertEqual(page.size, (48, 72))
            self.assertEqual(list(read_rows(render_pages(root, output, settings))), rows)
            settings["recursive"] = False
            self.assertEqual(len(list(read_rows(render_pages(root, output, settings)))), 2)

    @unittest.skipUnless(find_spec("fastapi"), "requires FastAPI")
    def test_console_resumes_interrupted_pdf_in_original_batch(self):
        from PIL import Image

        from rsi_ocr.manager import Manager

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            with Image.new("RGB", (12, 12), "white") as page:
                page.save(source, save_all=True, append_images=[page])
            settings = load_config("configs/import.yaml")
            settings["output_root"] = str(root / "runs")
            base = Path("configs/default.yaml")
            first = Manager(settings, base, 8520)
            job = {"value": str(source)}

            def interrupt(message):
                if "第 2/2 页" in message:
                    raise RuntimeError("测试中断")

            with patch.object(first, "report_progress", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "测试中断"):
                    first.infer(job)
            batch = Path(job["output"])
            page_one = batch / "input/pages/scan.pdf/1.png"
            self.assertTrue(page_one.is_file())
            self.assertFalse(page_one.with_name("2.png").exists())
            os.utime(page_one, (1, 1))

            second = Manager(settings, base, 8520)
            with patch.object(second, "infer_pages") as infer_pages:
                resumed = {"value": str(source)}
                second.infer(resumed)
                self.assertEqual(resumed["output"], str(batch))
                self.assertEqual(page_one.stat().st_mtime, 1)
                pages = infer_pages.call_args.args[2]
                self.assertEqual([row["page_id"] for row in read_rows(pages)], [1, 2])
                self.assertTrue(page_one.with_name("2.png").is_file())
                logs = [row["message"] for row in second.snapshot()["logs"]]
                self.assertIn("复用 PDF 页图 · 第 1/2 页 · 文件 1/1 · scan.pdf", logs)

                (batch / "production").mkdir()
                (batch / "production/summary.json").write_text("{}")
                fresh = {"value": str(source)}
                second.infer(fresh)
                self.assertNotEqual(fresh["output"], str(batch))

                with Image.new("RGB", (12, 12), "black") as page:
                    page.save(source)
                changed = {"value": str(source)}
                second.infer(changed)
                self.assertNotEqual(changed["output"], fresh["output"])

                settings["dpi"] *= 2
                resized = {"value": str(source)}
                second.infer(resized)
                self.assertNotEqual(resized["output"], changed["output"])
