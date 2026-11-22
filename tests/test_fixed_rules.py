import unittest

from rsi_ocr.config import load_config
from rsi_ocr.mining import FIXED_RULES, classify


class FixedRuleTests(unittest.TestCase):
    def test_rules_independently_toggle_without_hiding_evidence(self):
        policy = load_config("configs/default.yaml")["mining"]
        signals = {"target_risk": None, "disagreement": None, "judge_gain": None, "numeric_conflict": 1.0}
        for key in FIXED_RULES:
            target = {"status": key if key in {"empty", "format_invalid"} else "ok",
                      "judge": {"status": "precheck_reject"}}
            with self.subTest(rule=key):
                switches = dict.fromkeys(FIXED_RULES, False)
                off = classify(target, signals, {**policy, "fixed_rules": switches})
                self.assertFalse(off["hard_case"])
                self.assertEqual(off["signals"]["numeric_conflict"], 1.0)
                switches[key] = True
                on = classify(target, signals, {**policy, "fixed_rules": switches})
                self.assertEqual(on["reasons"], [key])
