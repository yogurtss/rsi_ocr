import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager, nullcontext
from threading import Event
from types import SimpleNamespace

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

    def test_render_and_layout_overlap_recognition_and_resume_by_page(self):
        from PIL import Image
        from rsi_ocr.capture import Recorder
        from rsi_ocr.importer import resume_batch
        from rsi_ocr.local import run_local
        from rsi_ocr.service import Service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            with Image.new("RGB", (18, 20), "white") as page:
                page.save(source, save_all=True, append_images=[page] * 5)
            settings = {"dpi": 72, "extensions": [".pdf"], "recursive": False, "output_root": root / "output"}
            config = load_config("configs/default.yaml")
            ocr_one = Event()
            layout_two = Event()
            render_three = Event()
            detections = []
            requests = []
            fail = True

            @contextmanager
            def listen(proxy):
                self.proxy = proxy
                yield "local-test"

            class Detector:
                def __call__(det, images, **kwargs):
                    batch = SimpleNamespace(instances=images, input_paths=[None], page_indexes=[None])
                    boxes = det.process(batch)["boxes"]
                    return [{"boxes": boxes[0]}]

                def process(det, batch):
                    number = batch.instances[0].number
                    detections.append(number)
                    if number == 2:
                        self.assertTrue(ocr_one.wait(3), "next layout must overlap current OCR")
                        layout_two.set()
                    return {"boxes": [[]], "input_img": batch.instances,
                            "input_path": batch.input_paths, "page_index": batch.page_indexes}

            inner = SimpleNamespace(layout_det_model=Detector(), img_reader=lambda paths: [
                SimpleNamespace(shape=(20, 18, 3), number=int(Path(path).stem)) for path in paths
            ])

            def predict(image, **kwargs):
                nonlocal fail
                result, = inner.layout_det_model([image])
                number = image.number
                self.assertEqual(self.proxy.page["page_id"], number)
                if number == 1:
                    ocr_one.set()
                    self.assertTrue(layout_two.wait(3), "first OCR must not block next layout")
                    self.assertTrue(render_three.wait(3), "PDF renderer must advance during OCR")
                    self.assertFalse((self.batch / "input/pages/scan.pdf/6.png").exists())
                if number == 2 and fail:
                    fail = False
                    raise RuntimeError("interrupt during OCR")
                # Identical crop bytes across pages must retain their own page and raw response.
                payload = {"model": "paddleocr-vl", "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "OCR:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
                ]}]}
                requests.append(number)
                self.proxy.forward("/chat/completions", payload)
                yield {"layout_det_res": result}

            pipeline = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=inner), predict_iter=predict)

            def progress(message):
                if message.startswith("PDF · 第 3/6 页"):
                    self.assertTrue(ocr_one.wait(3), "render must not drain the PDF before OCR starts")
                    render_three.set()

            def run(session=lambda *_: nullcontext()):
                with resume_batch(settings["output_root"], source, settings, lambda: None) as (batch, resumed, signatures):
                    self.batch = batch
                    source_plan = render_pages(source, batch / "input", settings, reuse_existing=resumed,
                                               signatures=signatures, streaming=True)
                    jobs = [(batch / "capture/regions.jsonl", batch / "production")]
                    run_local(config, batch, jobs, source_plan, stage="paddle", session=session,
                              progress=progress)

            def reply(*_):
                return {"choices": [{"message": {"content": f"OCR {requests[-1]}"}, "finish_reason": "stop"}]}

            module = SimpleNamespace(PaddleOCRVL=lambda **_: pipeline)
            with (patch.dict("sys.modules", {"paddleocr": module}), patch.object(Recorder, "listen", listen),
                  patch.object(Service, "request", side_effect=reply)):
                with self.assertRaisesRegex(RuntimeError, "interrupt during OCR"):
                    run()
                run()
                first_batch = self.batch
                (first_batch / "production/summary.json").write_text("{}")
                run(session=lambda *_: self.fail("no model on cache hit"))
                self.assertNotEqual(self.batch, first_batch)
                self.assertEqual((self.batch / "input/pages/scan.pdf/1.png").read_bytes(),
                                 (first_batch / "input/pages/scan.pdf/1.png").read_bytes())
            self.assertEqual(detections, list(range(1, 7)))
            self.assertEqual(requests, list(range(1, 7)))
            rows = list(read_rows(self.batch / "capture/regions.jsonl"))
            self.assertEqual([(r["page_id"], r["target"]["raw"]) for r in rows],
                             [(n, f"OCR {n}") for n in range(1, 7)])
            self.assertEqual(len(list(read_rows(self.batch / "input/pages.jsonl"))), 6)

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

            def consume(manager):
                return lambda job, output, pages: list(pages.images(manager.report_progress))

            with (patch.object(first, "report_progress", side_effect=interrupt),
                  patch.object(first, "infer_pages", side_effect=consume(first))):
                with self.assertRaisesRegex(RuntimeError, "测试中断"):
                    first.infer(job)
            batch = Path(job["output"])
            page_one = batch / "input/pages/scan.pdf/1.png"
            self.assertTrue(page_one.is_file())
            self.assertFalse(page_one.with_name("2.png").exists())
            os.utime(page_one, (1, 1))

            second = Manager(settings, base, 8520)
            with patch.object(second, "infer_pages", side_effect=consume(second)) as infer_pages:
                resumed = {"value": str(source)}
                second.infer(resumed)
                self.assertEqual(resumed["output"], str(batch))
                self.assertEqual(page_one.stat().st_mtime, 1)
                pages = infer_pages.call_args.args[2].manifest
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
