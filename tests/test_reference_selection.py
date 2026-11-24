import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rsi_ocr.benchmark import evaluate, reference_comparison
from rsi_ocr.store import read_rows, write_rows


class ReferenceSelectionTests(unittest.TestCase):
    def test_references_only_audit_the_saved_choice(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.jsonl"
            refs = root / "references.jsonl"
            key = {"doc_id": "a", "page_id": 1, "region_id": "2", "task": "text", "image_sha256": "crop"}
            row = {**key, "case_id": "c", "target_alias": "paddle", "selection": {"model_alias": "oxr"},
                   "target_operational_error": False, "hard_case": True, "priority": 0.8,
                   "models": {"paddle": {"status": "ok", "label": "12"}, "oxr": {"status": "ok", "label": "21"}}}
            write_rows(source, [row, {**row, "case_id": "unlabelled", "region_id": "3"}])
            saved = source.read_bytes()
            for truth, winner in [("12", "paddle"), ("21", "oxr")]:
                write_rows(refs, [{**key, "reference": truth}])
                report = evaluate(source, refs, root / "evaluation", 25, "all")
                case, = read_rows(root / "evaluation/cases.jsonl")
                self.assertEqual(case["selection_gt"]["selected_alias"], "oxr")
                self.assertEqual(case["selection_gt"]["closest_aliases"], [winner])
                self.assertEqual(report["selection_gt"], {"with_reference": 1, "comparable": 1,
                                                         "closest": int(winner == "oxr")})
                self.assertEqual(report["unmatched"], 1)
                self.assertEqual(source.read_bytes(), saved)

    def test_teds_ties_and_absent_selection(self):
        row = {"task": "table", "models": {alias: {"status": "ok", "label": alias} for alias in ("a", "b", "c")},
               "selection": {"model_alias": "b"}}
        metrics = {"a": {"teds": 0.8}, "b": {"teds": 0.8}, "c": {"teds": 0.5}}
        audit = reference_comparison(row, metrics)
        self.assertEqual(audit["closest_aliases"], ["a", "b"])
        self.assertTrue(audit["is_closest"])
        row["selection"]["model_alias"] = None
        self.assertIsNone(reference_comparison(row, metrics)["is_closest"])
