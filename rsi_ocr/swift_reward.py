import json
import os

from swift.rewards import ORM, orms

from rsi_ocr.reward import score, validate_weights


class RSIOCRReward(ORM):
    def __init__(self):
        self.config = json.loads(os.environ["RSI_OCR_REWARD_WEIGHTS"])
        validate_weights(self.config)

    def __call__(self, completions, solution, ocr_task, **kwargs):
        return [score(task, prediction, answer, self.config)["score"]
                for task, prediction, answer in zip(ocr_task, completions, solution, strict=True)]


orms["rsi_ocr_region"] = RSIOCRReward
