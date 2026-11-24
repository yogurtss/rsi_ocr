"""Batch progress in page and region units, shared by the console and monitor."""

STAGE_NAMES = {"render": "页图", "paddle": "Paddle", "oxr": "OXR", "ovis": "Ovis", "jev": "Jev", "mine": "筛选"}


def fraction(stage, doc=None):
    if doc is None:
        done, total = stage["processed"], stage["total"]
    else:
        done, total, _ = stage["docs"].get(doc, (0, 0, 0))
    if total:
        return done / total
    return float(stage["planned"] and (doc is not None or stage["finished"]))


def overall(stages, doc=None):
    if not stages:
        return 0
    value = sum(fraction(stage, doc) for stage in stages.values()) / len(stages)
    if doc is None and not all(stage["finished"] for stage in stages.values()):
        return min(value, 0.99)
    return value


class BatchProgress:
    """The manager lock serializes all updates, including concurrent teacher stages."""

    def __init__(self, teachers, judge):
        self.stages = {name: {"planned": False, "finished": False, "processed": 0, "total": 0,
                              "errors": 0, "docs": {}, "current": None}
                       for name in ("render", "paddle", *teachers, judge or "mine")}
        self.current_doc = None

    def plan(self, name, totals, completed):
        stage = self.stages[name]
        stage["planned"] = True
        for doc, total in totals.items():
            done = completed.get(doc, 0)
            stage["docs"][doc] = (done, total, 0)
        stage["total"] = sum(totals.values())
        stage["processed"] = sum(completed.values())

    def advance(self, name, doc, failed):
        stage = self.stages[name]
        done, total, errors = stage["docs"][doc]
        stage["docs"][doc] = (done + 1, total, errors + failed)
        stage["processed"] += 1
        stage["errors"] += failed
        stage["current"] = self.current_doc = doc

    def snapshot(self):
        return {"current_doc": self.current_doc,
                "stages": {name: {**stage, "docs": stage["docs"].copy()} for name, stage in self.stages.items()}}
