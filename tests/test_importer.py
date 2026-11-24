import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

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
