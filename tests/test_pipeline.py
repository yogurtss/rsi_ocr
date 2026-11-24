import base64
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from rsi_ocr.capture import Recorder, capture_pages
from rsi_ocr.config import load_config
from rsi_ocr.formats import table_cells, table_otsl
from rsi_ocr.mining import mine
from rsi_ocr.reward import score
from rsi_ocr.service import Service
from rsi_ocr.store import digest, read_rows, write_rows
from rsi_ocr.training import export_training, train_command

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR1sAAAAASUVORK5CYII=")
HTML = '<table><tr><td colspan="2">Item</td></tr><tr><td>001</td><td>-10.50</td></tr></table>'
MERGED = '<table><tr><td rowspan="2" colspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'


def chat(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "block.png").write_bytes(PNG)

    def test_reward_penalizes_numeric_errors_spans_and_wrong_format(self):
        config = self.config["reward"]
        self.assertEqual(score("text", "  编号 001 −10.50% ", "编号 001 −10.50%", config)["score"], 1)
        self.assertLess(score("text", "编号 001 10.50%", "编号 001 −10.50%", config)["score"], 0.85)
        self.assertEqual(score("text", "", "test", config)["score"], 0)
        otsl = table_otsl(table_cells(MERGED))
        self.assertEqual(table_cells(otsl), table_cells(MERGED))
        self.assertEqual(score("table", otsl, MERGED, config)["score"], 1)
        self.assertEqual(score("table", MERGED, MERGED, config)["score"], 0)
        self.assertEqual(score("table", otsl.replace("<xcel>", "<lcel>"), MERGED, config)["score"], 0)
        otsl = table_otsl(table_cells(HTML))
        wrong_number = score("table", otsl.replace("-10.50", "10.50"), HTML, config)
        self.assertEqual(wrong_number["structure"], 1)
        self.assertLess(wrong_number["numeric"], 1)
        self.assertLess(wrong_number["score"], 1)

    def test_capture_records_unchanged_crop_and_resumes(self):
        self.config["capture"]["pipeline"]["pipeline_version"] = "v1.6"
        module = ModuleType("paddleocr")
        test = self

        @contextmanager
        def local_transport(recorder):
            test.recorder = recorder
            yield "local-test-transport"

        class PaddlePipeline:
            def __init__(self, **options):
                self.url = options["vl_rec_server_url"]

            def predict_iter(self, page, **options):
                test.assertTrue(options["use_layout_detection"])
                content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
                           {"type": "text", "text": "OCR:"}]
                payload = {"model": "paddleocr-vl", "messages": [{"role": "user", "content": content}], "max_tokens": 200}
                status, result = test.recorder.forward("/chat/completions", payload)
                test.assertEqual(status, 200)
                yield result

        module.PaddleOCRVL = PaddlePipeline
        pages = self.root / "pages.jsonl"
        write_rows(pages, [{"doc_id": "d", "page_id": 0, "image": "block.png"}])
        with (patch.dict("sys.modules", {"paddleocr": module}), patch.object(Recorder, "listen", local_transport),
              patch.object(Service, "request", return_value=chat("raw OCR")) as upstream):
            path = capture_pages(self.config, pages, self.root / "capture")
            progress = []
            capture_pages(self.config, pages, self.root / "capture", progress=progress.append)
            upstream.assert_called_once()
        self.assertEqual(progress, ["初始化 PaddleOCR 版面识别", "复用 OCR · 第 1/1 页 · 文件 1/1 · d"])
        row, = read_rows(path)
        self.assertEqual((path.parent / row["image"]).read_bytes(), PNG)
        self.assertEqual(row["target"]["raw"], "raw OCR")
        self.assertEqual(row["target"]["parameters"], {"max_tokens": 200})

    def test_mining_verdicts_audit_and_approved_exports(self):
        region_file = self.root / "regions.jsonl"
        write_rows(region_file, [
            {"doc_id": "d", "page_id": 0, "region_id": "t", "image": "block.png", "task": "text"},
            {"doc_id": "d", "page_id": 0, "region_id": "b", "image": "block.png", "task": "table"},
            {"doc_id": "holdout", "page_id": 0, "region_id": "held", "image": "block.png", "task": "text", "split": "test"},
        ])
        self.config["mining"]["audit_rate"] = 0
        self.config["mining"]["workers"] = 2
        output = self.root / "mine"
        otsl = table_otsl(table_cells(HTML))

        def request(service, path, payload):
            if path.endswith("/score"):
                items = []
                for item in payload["items"]:
                    self.assertEqual(base64.b64decode(item["image"]), PNG)
                    if item["category"] == "table":
                        items.append({"verdict": "render_error", "prob_yes": 0.0})
                    else:
                        items.append({"verdict": "scored", "prob_yes": 0.95 if "001" in item["ocr_text"] else 0.05})
                return {"results": items}
            model = service.config
            prompt = payload["messages"][0]["content"][1]["text"]
            task = next(name for name, spec in model["tasks"].items() if spec["prompt"] == prompt)
            primary = model["model"] == "paddleocr-vl"
            if task == "text":
                return chat("Total 1" if primary else "Total 001")
            return chat(otsl.replace("-10.50", "10.50") if primary else HTML)

        with patch.object(Service, "request", autospec=True, side_effect=request) as upstream:
            summary = mine(self.config, region_file, output)
            calls = upstream.call_count
            mine(self.config, region_file, output)
            self.assertEqual(upstream.call_count, calls + 1)
        self.assertEqual(summary["scanned"], 2)
        rows = {row["task"]: row for row in read_rows(output / "review.jsonl")}
        self.assertIn("numeric_conflict", rows["text"]["reasons"])
        self.assertTrue(rows["text"]["proposals"][0]["recommended"])
        self.assertIsNone(rows["table"]["models"]["paddle"]["judge"]["prob_yes"])
        self.assertEqual(rows["table"]["models"]["paddle"]["judge"]["status"], "render_error")
        decisions = self.root / "decisions.jsonl"
        write_rows(decisions, [{"case_id": rows["table"]["case_id"], "image_sha256": digest(PNG), "status": "approved",
                                "source": "human_verified", "label": HTML}])
        data = self.root / "data"
        self.assertEqual(export_training(self.config, output / "evidence.jsonl", decisions, data), {"train": 1, "val": 0})
        sft, = read_rows(data / "sft_train.jsonl")
        self.assertEqual(sft["text_info"][1], {"text": otsl, "tag": "no_mask"})
        grpo, = read_rows(data / "grpo_train.jsonl")
        self.assertEqual(grpo["ocr_task"], "table")
        self.assertEqual(score("table", otsl, grpo["solution"], self.config["reward"])["score"], 1)
        with patch.dict(os.environ, {"ERNIEKIT_REPO": str(self.root)}):
            args, _, cwd = train_command(self.config, "sft", data, self.root / "sft", None)
        self.assertEqual(cwd, self.root)
        self.assertIn(f"train_dataset_path={data / 'sft_train.jsonl'}", args)
        args, env, _ = train_command(self.config, "grpo", data, self.root / "grpo", None)
        self.assertIn("rsi_ocr_region", args)
        self.assertEqual(json.loads(env["RSI_OCR_REWARD_WEIGHTS"]), self.config["reward"])

    def test_local_batch_mines_without_teachers_or_jev(self):
        from rsi_ocr.importer import runtime_config
        from rsi_ocr.local import run_local

        settings = load_config(DEFAULT.with_name("import.yaml"))
        for alias in ("oxr", "ovis", "jev"):
            settings["deployment"]["models"][alias]["enabled"] = False
        config = runtime_config(settings, DEFAULT)
        manifest = self.root / "regions.jsonl"
        write_rows(manifest, [
            {"doc_id": "d", "page_id": 0, "region_id": "t", "image": "block.png", "task": "text",
             "target": {"status": "ok", "raw": "Paddle output", "format": "text"}},
        ])
        started = []

        @contextmanager
        def session(alias, output):
            started.append(alias)
            yield

        output = self.root / "production"
        with patch.object(Service, "request") as upstream:
            run_local(config, self.root, [(manifest, output)], session=session)
            upstream.assert_not_called()
        self.assertEqual(started, ["paddle"])
        row, = read_rows(output / "evidence.jsonl")
        self.assertEqual(set(row["models"]), {"paddle"})
        self.assertEqual(row["models"]["paddle"]["raw"], "Paddle output")
        self.assertIsNone(row["priority"])
