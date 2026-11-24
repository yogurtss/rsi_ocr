"""Serve the local OCR review workbench."""
import argparse
from collections import Counter
import difflib
import json
import re
import sqlite3
from typing import Literal
from uuid import uuid4
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from rsi_ocr.mining import FIXED_RULES, classify, select
from rsi_ocr.formats import table_cells, table_metrics
from rsi_ocr.store import digest, json_text, read_rows, write_rows
from rsi_ocr.training import paddle_sft_record

ASSETS = Path(__file__).with_name("review_ui")
SFT_SPLITS = {"mine": "train", "train": "train", "val": "val"}


def decision_db(path):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS decisions (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS review_events (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS review_locations (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS batch_policy (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    return db


def decisions(path):
    with closing(decision_db(path)) as db:
        return {key: json.loads(value) for key, value in db.execute("SELECT case_id, payload FROM decisions")}


def verified_image(row, record):
    image = Path(row["image"]).resolve()
    if (record["image_sha256"] != row["image_sha256"] or not image.is_file()
            or digest(image.read_bytes()) != row["image_sha256"]):
        raise ValueError(f"裁图已丢失或发生变化（区域 {row['region_id']}）")
    return image


def decision_record(row, status, alias):
    model = row["models"].get(alias, {})
    label = (model.get("label") or "").strip() if status == "approved" else ""
    record = {"case_id": row["case_id"], "image_sha256": row["image_sha256"],
              "status": status, "source": "human_verified", "label": label, "model_alias": alias,
              "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
    if status == "approved":
        if not label:
            raise ValueError("请选择一条可用的模型转录，再点 Pass")
        image = verified_image(row, record)
        if row.get("split", "mine") in SFT_SPLITS:
            record["sft"] = paddle_sft_record(image, row["task"], label)
    return record


def store_record(db, record):
    payload = json_text(record)
    db.execute("INSERT INTO decisions VALUES (?, ?) ON CONFLICT(case_id) DO UPDATE SET payload=excluded.payload",
               (record["case_id"], payload))
    db.execute("INSERT INTO review_events (payload) VALUES (?)", (payload,))


def save_decision(path, row, status, alias):
    record = decision_record(row, status, alias)
    with closing(decision_db(path)) as db, db:
        store_record(db, record)
    return record


def sft_exports(rows, records):
    datasets = {"train": [], "val": []}
    seen = {}
    for record in records.values():
        row = rows.get(record["case_id"])
        if row is None or record["status"] != "approved":
            continue
        split = SFT_SPLITS.get(row.get("split", "mine"))
        if split is None:
            continue
        image = verified_image(row, record)
        sample = paddle_sft_record(image, row["task"], record["label"])
        key = row["task"], row["image_sha256"]
        value = split, sample["text_info"][1]["text"]
        if key in seen:
            if seen[key] != value:
                raise ValueError(f"同一裁图存在分区或标注冲突（区域 {row['region_id']}）")
            continue
        seen[key] = value
        datasets[split].append(sample)
    return datasets


@lru_cache(maxsize=1024)
def match_crop(page_path, crop_path, page_mtime, crop_mtime):
    import cv2

    page = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if page is None or crop is None or crop.std() < 5:
        return None
    height, width = crop.shape
    if height > page.shape[0] or width > page.shape[1]:
        return None
    match = cv2.matchTemplate(page, crop, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(match)
    if best < 0.985:
        return None
    match[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = -1
    if best - float(match.max()) < 0.025:
        return None
    return x, y, x + width, y + height


def page_geometry(row, manual_box=None):
    page = Path(row["page_image"])
    with Image.open(page) as source:
        width, height = source.size
    box = manual_box or row.get("bbox")
    crop = Path(row["image"])
    if not box and crop.is_file():
        box = match_crop(page, crop, page.stat().st_mtime_ns, crop.stat().st_mtime_ns)
    if box:
        left, top, right, bottom = box
        box = max(0, left), max(0, top), min(width, right), min(height, bottom)
        if box[0] >= box[2] or box[1] >= box[3]:
            box = None
    return {"width": width, "height": height, "box": box,
            "location": ("manual" if manual_box else "recorded" if row.get("bbox") else "matched") if box else None}


def document(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_batch(run, evaluation, title):
    pages = run.parent / "input" / "pages.jsonl"
    sources = Counter(row["doc_id"] for row in read_rows(pages)) if pages.is_file() else {}
    return {
        "sources": sources,
        "title": title, "path": run.resolve(),
        "summary": document(run / "summary.json"),
        "rows": {row["case_id"]: row for row in read_rows(run / "evidence.jsonl")},
        "queue": {row["case_id"] for row in read_rows(run / "review.jsonl")},
        "details": {row["case_id"]: row for row in read_rows(evaluation / "cases.jsonl")} if evaluation else {},
        "report": document(evaluation / "report.json") if evaluation else None,
    }


def table_view(label, baseline):
    cells = table_cells(label)
    comparison = table_cells(baseline) if baseline else []
    expected = {cell.position: cell.text for cell in comparison}
    rows = [[] for _ in range(max(cell.row + cell.height for cell in cells))]
    for cell in cells:
        rows[cell.row].append({
            "text": cell.text, "col": cell.col, "width": cell.width, "height": cell.height,
            "changed": bool(comparison) and expected.get(cell.position) != cell.text,
        })
    metrics = table_metrics(cells, comparison) if comparison else None
    return {"rows": rows, "structure": metrics[0] if metrics else None,
            "content": metrics[1] if metrics else None, "numeric": metrics[2] if metrics else None}


def output_view(label, baseline, task):
    if not label:
        return {}
    if task == "table":
        try:
            return table_view(label, baseline)
        except ValueError:
            return {}
    if not baseline or label == baseline:
        return {"parts": [["equal", label]]}
    matcher = difflib.SequenceMatcher(None, baseline, label, autojunk=max(len(label), len(baseline)) > 12000)
    parts = []
    for tag, left, right, start, stop in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            parts.append(["delete", baseline[left:right]])
        if start != stop:
            parts.append(["equal" if tag == "equal" else "insert", label[start:stop]])
    return {"parts": parts}


class FixedRules(BaseModel):
    numeric_conflict: bool = True
    empty: bool = True
    format_invalid: bool = True
    judge_precheck_reject: bool = True


class Thresholds(BaseModel):
    low_judge: float = Field(ge=0, le=1)
    disagreement: float = Field(ge=0, le=1)
    judge_gain: float = Field(ge=0, le=1)
    fixed_rules: FixedRules | None = None


def threshold_values(policy):
    values = {key: policy[key] for key in ("low_judge", "disagreement", "judge_gain")}
    values["fixed_rules"] = {key: policy.get("fixed_rules", {}).get(key, True) for key in FIXED_RULES}
    return values


def project_batch(batch, thresholds):
    policy = {**batch["summary"]["mining"], **thresholds}
    rows = {}
    for key, row in batch["rows"].items():
        result = classify(row["models"][row["target_alias"]], row["signals"], policy)
        rows[key] = {**row, **result}
    queue = {row["case_id"] for row in select(rows.values(), policy)}
    summary = {**batch["summary"], "mining": policy, "review": len(queue),
               "hard_cases": sum(row["hard_case"] for row in rows.values())}
    return {**batch, "rows": rows, "queue": queue, "summary": summary}


class Decision(BaseModel):
    status: str
    model_alias: str


class Location(BaseModel):
    box: tuple[int, int, int, int]


def create_app(run=None, evaluation=None, production=None, library=None, manager_port=None, *, host="127.0.0.1"):
    batches = {}
    if run:
        title = "OmniDocBench 基准评测" if evaluation else run.name
        batches["benchmark"] = read_batch(run, evaluation, title)
    if production:
        batches["production"] = read_batch(production, None, "Paddle 原页采集")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    if host not in {"0.0.0.0", "::"}:
        authority = f"[{host}]" if ":" in host else host
        allowed = re.compile(rf"(?:127\.0\.0\.1|localhost|{re.escape(authority)})(?::[0-9]+)?", re.IGNORECASE)

        @app.middleware("http")
        async def check_host(request: Request, call_next):
            if not allowed.fullmatch(request.headers.get("host", "")):
                return PlainTextResponse("Invalid host header", status_code=400)
            return await call_next(request)

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

    @app.get("/api/runtime")
    def runtime():
        if manager_port is None:
            return {"connected": None}
        from .pageproof import endpoint
        try:
            status = endpoint(manager_port, "/status")
        except ValueError:
            return {"connected": False}
        layout_device = status.get("layout_device", "cpu")
        return {"connected": True, "mode": status.get("mode"), "models": status["models"],
                "layout": status.get("layout") or {"path": "PP-DocLayoutV3", "device": layout_device},
                "layout_device": layout_device,
                "jobs": [job for job in status["jobs"] if job["status"] in {"running", "queued"}]}

    @app.put("/api/runtime/deployment/{mode}")
    def change_mode(mode: Literal["on_demand", "managed"]):
        if manager_port is None:
            raise HTTPException(409, "此工作台未连接模型后台")
        from .pageproof import endpoint
        try:
            return endpoint(manager_port, "/deployment/" + mode, {})
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.put("/api/runtime/deployment")
    def save_deployment(settings: dict):
        if manager_port is None:
            raise HTTPException(409, "此工作台未连接模型后台")
        from .pageproof import endpoint
        try:
            return endpoint(manager_port, "/deployment", settings, timeout=120)
        except ValueError as error:
            detail = str(error)
            raise HTTPException(409, detail) from error

    def original_batch(batch_id):
        if batch_id not in batches:
            raise HTTPException(404, "批次不存在")
        return batches[batch_id]

    def active_policy(batch, db):
        saved = db.execute("SELECT payload FROM batch_policy WHERE id=1").fetchone()
        if not saved:
            return batch
        cached = batch.get("policy_cache")
        if cached and cached[0] == saved[0]:
            return cached[1]
        projected = project_batch(batch, json.loads(saved[0]))
        projected.pop("policy_cache", None)
        batch["policy_cache"] = (saved[0], projected)
        return projected

    def get_batch(batch_id):
        batch = original_batch(batch_id)
        with closing(decision_db(batch["path"] / "review.sqlite")) as db:
            return active_policy(batch, db)

    @app.post("/api/batches/{batch_id}/policy/preview")
    def preview_policy(batch_id: str, thresholds: Thresholds):
        batch = project_batch(original_batch(batch_id), thresholds.model_dump(exclude_none=True))
        hard = [row for row in batch["rows"].values() if row["hard_case"]]
        return {"hard": len(hard), "queued": len(batch["queue"]),
                "by_task": Counter(row["task"] for row in hard),
                "reasons": Counter(reason for row in hard for reason in row["reasons"])}

    @app.put("/api/batches/{batch_id}/policy")
    def apply_policy(batch_id: str, thresholds: Thresholds):
        batch = original_batch(batch_id)
        values = thresholds.model_dump(exclude_none=True)
        with closing(decision_db(batch["path"] / "review.sqlite")) as db, db:
            if values == threshold_values(batch["summary"]["mining"]):
                db.execute("DELETE FROM batch_policy WHERE id=1")
            else:
                db.execute("INSERT OR REPLACE INTO batch_policy VALUES (1, ?)", (json_text(values),))
        return values

    def get_case(batch_id, case_id):
        batch = get_batch(batch_id)
        if case_id not in batch["rows"]:
            raise HTTPException(404, "样本不存在")
        return batch, batch["rows"][case_id]

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html", headers={"Cache-Control": "no-cache"})

    def available_runs():
        if not library or not library.is_dir():
            return {}
        result = {}
        for path in sorted(library.iterdir(), reverse=True):
            run_path = path / "production"
            if run_path.resolve().is_relative_to(library.resolve()) and all(
                (run_path / name).is_file() for name in ("summary.json", "review.jsonl", "evidence.jsonl")
            ):
                result[path.name] = run_path
        return result

    @app.get("/api/imports")
    def list_imports():
        return {"directory": str(library or ""), "names": list(available_runs())}

    @app.post("/api/imports/{name}")
    def import_batch(name: str):
        path = available_runs().get(name)
        if path is None:
            raise HTTPException(404, "批次不存在或仍在生成")
        key = "import-" + digest(name.encode())[:20]
        try:
            batches[key] = read_batch(path, None, name)
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(422, f"批次读取失败：{error}") from error
        return {"id": key, "name": name}

    @app.get("/api/batches")
    def batch_list():
        return {key: batch["title"] for key, batch in batches.items()}

    @app.get("/api/batches/{batch_id}")
    def batch_data(batch_id: str):
        batch = get_batch(batch_id)
        recorded = decisions(batch["path"] / "review.sqlite")
        rows = []
        for case_id, row in batch["rows"].items():
            target = row["models"][row["target_alias"]]
            detail = batch["details"].get(case_id, {})
            rows.append({
                **{key: row.get(key) for key in
                   ("case_id", "doc_id", "page_id", "region_id", "task", "priority", "reasons", "audit_sample", "hard_case")},
                "operational": bool(row.get("target_operational_error")),
                "valid_target": bool(target.get("label")),
                "queued": case_id in batch["queue"], "judge_status": target.get("judge", {}).get("status"),
                "review_status": recorded.get(case_id, {}).get("status", "pending"),
                "target_error": detail.get("target_error"), "repairable": detail.get("repairable"),
                "exportable": row.get("split", "mine") in SFT_SPLITS,
                "split": row.get("split", "mine"),
            })
        return {"rows": rows, "summary": batch["summary"], "report": batch["report"], "sources": batch["sources"],
                "original_thresholds": threshold_values(original_batch(batch_id)["summary"]["mining"])}

    @app.post("/api/batches/{batch_id}/bulk")
    def bulk_pass(batch_id: str, ids: list[str]):
        batch = original_batch(batch_id)
        token = uuid4().hex
        changed = []
        with closing(decision_db(batch["path"] / "review.sqlite")) as db, db:
            db.execute("BEGIN IMMEDIATE")
            batch = active_policy(batch, db)
            for key in dict.fromkeys(ids):
                row = batch["rows"].get(key)
                if not row or row.get("hard_case") is not False or row.get("target_operational_error"):
                    continue
                if db.execute("SELECT 1 FROM decisions WHERE case_id=?", (key,)).fetchone():
                    continue
                if not row["models"][row["target_alias"]].get("label"):
                    continue
                try:
                    record = decision_record(row, "approved", row["target_alias"])
                except (ValueError, OSError) as error:
                    raise HTTPException(422, str(error)) from error
                record.update(source="bulk_approved", bulk_id=token)
                store_record(db, record)
                changed.append(key)
        return {"bulk_id": token, "changed": changed}

    @app.post("/api/batches/{batch_id}/bulk/{token}/undo")
    def undo_bulk(batch_id: str, token: str):
        batch = get_batch(batch_id)
        changed = []
        with closing(decision_db(batch["path"] / "review.sqlite")) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for key, value in db.execute("SELECT case_id,payload FROM decisions WHERE json_extract(payload,'$.bulk_id')=?", (token,)).fetchall():
                db.execute("DELETE FROM decisions WHERE case_id=?", (key,))
                event = dict(json.loads(value), status="undone", timestamp=datetime.now(timezone.utc).isoformat())
                db.execute("INSERT INTO review_events (payload) VALUES (?)", (json_text(event),))
                changed.append(key)
        return {"changed": changed}

    @app.get("/api/batches/{batch_id}/cases/{case_id}/comparison/{alias}")
    def compare_with(batch_id: str, case_id: str, alias: str):
        _, row = get_case(batch_id, case_id)
        if alias not in row["models"]:
            raise HTTPException(404, "模型不存在")
        base = row["models"][alias].get("label")
        return {key: output_view(model.get("label"), base, row["task"]) for key, model in row["models"].items()}

    @app.get("/api/batches/{batch_id}/cases/{case_id}")
    def case_data(batch_id: str, case_id: str):
        batch, row = get_case(batch_id, case_id)
        detail = batch["details"].get(case_id, {})
        reference = detail.get("reference")
        target_label = row["models"][row["target_alias"]].get("label")
        views = {alias: output_view(model.get("label"), reference or target_label, row["task"])
                 for alias, model in row["models"].items()}
        baseline_views = ({alias: output_view(model.get("label"), target_label, row["task"])
                           for alias, model in row["models"].items()} if reference else views)
        has_page = bool(row.get("page_image") and Path(row["page_image"]).is_file())
        database = batch["path"] / "review.sqlite"
        with closing(decision_db(database)) as db:
            saved_location = db.execute("SELECT payload FROM review_locations WHERE case_id = ?", (case_id,)).fetchone()
        manual_box = json.loads(saved_location[0])["box"] if saved_location else None
        return {
            **row, "metrics": detail.get("models", {}), "reference": reference,
            "reference_view": output_view(reference, None, row["task"]), "views": views,
            "baseline_views": baseline_views,
            "has_crop": Path(row["image"]).is_file(),
            "page_geometry": page_geometry(row, manual_box) if has_page else None,
            "export_split": SFT_SPLITS.get(row.get("split", "mine")),
            "decision": decisions(batch["path"] / "review.sqlite").get(case_id),
            "policy": batch["summary"]["mining"],
        }

    @app.get("/api/batches/{batch_id}/cases/{case_id}/image/{kind}")
    def case_image(batch_id: str, case_id: str, kind: str):
        _, row = get_case(batch_id, case_id)
        field = {"crop": "image", "page": "page_image"}.get(kind)
        if not field or not row.get(field) or not Path(row[field]).is_file():
            raise HTTPException(404, "原图不可用")
        return FileResponse(row[field], headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/batches/{batch_id}/cases/{case_id}/decision")
    def record_decision(batch_id: str, case_id: str, decision: Decision):
        batch, row = get_case(batch_id, case_id)
        if decision.status not in {"approved", "rejected"} or decision.model_alias not in row["models"]:
            raise HTTPException(422, "复核状态无效")
        path = batch["path"] / "review.sqlite"
        try:
            return save_decision(path, row, decision.status, decision.model_alias)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.put("/api/batches/{batch_id}/cases/{case_id}/location")
    def record_location(batch_id: str, case_id: str, location: Location):
        batch, row = get_case(batch_id, case_id)
        if not row.get("page_image") or not Path(row["page_image"]).is_file():
            raise HTTPException(404, "原页文件不可用")
        geometry = page_geometry(row, location.box)
        if geometry["box"] != location.box or geometry["location"] != "manual":
            raise HTTPException(422, "请在原页范围内框出区域")
        record = {"box": location.box, "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
        with closing(decision_db(batch["path"] / "review.sqlite")) as db, db:
            db.execute("INSERT INTO review_locations VALUES (?, ?) ON CONFLICT(case_id) DO UPDATE SET payload=excluded.payload",
                       (case_id, json_text(record)))
        return geometry

    @app.get("/api/batches/{batch_id}/export/review")
    def export_decisions(batch_id: str):
        batch = get_batch(batch_id)
        records = decisions(batch["path"] / "review.sqlite")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        # Keep the audit JSONL compatible with the existing CLI exporter.
        return PlainTextResponse("".join(json_text({key: value for key, value in row.items() if key != "sft"}) + "\n"
                                         for row in records.values()),
                                 headers={"Content-Disposition": f'attachment; filename="review_{stamp}.jsonl"'})

    @app.post("/api/batches/{batch_id}/export/sft/{split}")
    def export_sft(batch_id: str, split: str):
        if split not in {"train", "val"}:
            raise HTTPException(404, "导出分区不存在")
        batch = get_batch(batch_id)
        try:
            samples = sft_exports(batch["rows"], decisions(batch["path"] / "review.sqlite"))[split]
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if not samples:
            raise HTTPException(409, "该分区还没有通过复核的训练样本")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = batch["path"] / "sft_exports" / f"sft_{split}_{stamp}.jsonl"
        write_rows(path, samples)
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    return app


def main():
    import uvicorn

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path)
    cli.add_argument("--evaluation", type=Path)
    cli.add_argument("--production", type=Path)
    cli.add_argument("--settings", type=Path, default=Path("configs/import.yaml"))
    cli.add_argument("--port", type=int)
    cli.add_argument("--host", "-host", default="127.0.0.1", help="工作台监听地址，默认 127.0.0.1")
    cli.add_argument("--manager-port", type=int, help="connect deployment settings to a local model manager")
    args = cli.parse_args()
    from .config import load_config
    settings = load_config(args.settings)
    app = create_app(args.run, args.evaluation, args.production, Path(settings["output_root"]).resolve(),
                     args.manager_port, host=args.host)
    uvicorn.run(app, host=args.host, port=args.port or (8516 if args.run else settings["port"]))


if __name__ == "__main__":
    main()
