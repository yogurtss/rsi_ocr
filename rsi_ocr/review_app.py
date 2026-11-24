"""Serve the local OCR review workbench."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
import re
import sqlite3
from threading import Lock
from time import monotonic
from typing import Literal
from uuid import uuid4
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from itertools import islice
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from rsi_ocr.benchmark import reference_comparison, reference_metrics, region_key, selection_summary
from rsi_ocr.mining import FIXED_RULES, automatic_label, stored_selection, classify, select, selected_result
from rsi_ocr.formats import table_cells, table_metrics
from rsi_ocr.store import digest, json_text, write_rows
from rsi_ocr.training import paddle_sft_record

ASSETS = Path(__file__).with_name("review_ui")
SFT_SPLITS = {"mine": "train", "train": "train", "val": "val"}
BATCH_FILES = ("summary.json", "review.jsonl", "evidence.jsonl")


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
            raise ValueError("请选择一条可用的模型转录，再点「确认通过」")
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


def sft_exports(batch, records):
    datasets = {"train": [], "val": []}
    seen = {}
    with evidence_stream(batch) as stream:
        for row in batch["rows"].values():
            record = records.get(row["case_id"])
            if record and record["status"] != "approved":
                continue
            if not record:
                if not row["selection"].get("model_alias") or row.get("split", "mine") not in SFT_SPLITS:
                    continue
                record = automatic_label(evidence_row(stream, row), row["selection"])
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


@lru_cache(maxsize=48)
def page_thumbnail(path, mtime_ns):
    with Image.open(path, formats=("JPEG", "PNG")) as page, BytesIO() as buffer:
        page.thumbnail((560, 360), Image.Resampling.LANCZOS)
        page.convert("RGB").save(buffer, "JPEG", quality=82)
        return buffer.getvalue()


def import_cover(run):
    directory = run.parent / "input"
    with (directory / "pages.jsonl").open(encoding="utf-8-sig") as stream:
        first = json.loads(stream.readline())
    image = (directory / first["image"]).resolve()
    cover = image if image.is_relative_to(run.parent.resolve() / "input" / "pages") and image.is_file() else None
    return first.get("doc_id"), cover


def file_version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def import_version(run):
    inputs = run.parent / "input"
    version = []
    paths = [run / name for name in BATCH_FILES]
    for path in paths + [inputs / "pages.jsonl", inputs / "references.jsonl"]:
        try:
            version.append(file_version(path.stat()))
        except FileNotFoundError:
            version.append(None)
    return tuple(version)


def batch_lines(path, stage, report):
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        offset = 0
        records = 0
        if report:
            report(stage, path.name, records, offset, size)
        next_update = monotonic() + 0.25
        for line in stream:
            if report and monotonic() >= next_update:
                report(stage, path.name, records, offset, size)
                next_update = monotonic() + 0.25
            start = offset
            offset += len(line)
            if start == 0:
                line = line.removeprefix(b"\xef\xbb\xbf")
            if line.strip():
                yield start, line
                records += 1
        if report:
            report(stage, path.name, records, offset, size)


@contextmanager
def evidence_stream(batch):
    try:
        stream = (batch["path"] / "evidence.jsonl").open("rb")
    except OSError as error:
        raise HTTPException(409, "无法读取批次的 evidence.jsonl，请检查目录并重新导入") from error
    with stream:
        if file_version(os.fstat(stream.fileno())) != batch["evidence_version"]:
            raise HTTPException(409, "识别结果已更新，请重新导入该批次")
        yield stream


def evidence_row(stream, entry):
    stream.seek(entry["_offset"])
    row = json.loads(stream.readline())
    if row["case_id"] != entry["case_id"]:
        raise HTTPException(409, "识别结果已更新，请重新导入该批次")
    for key in ("selection", "reasons", "priority", "hard_case", "target_operational_error"):
        if key in entry:
            row[key] = entry[key]
    return row


def read_batch(run, evaluation, title, report=None):
    pages = run.parent / "input" / "pages.jsonl"
    sources = Counter()
    if pages.is_file():
        sources.update(json.loads(line)["doc_id"] for _, line in batch_lines(pages, "读取页清单", report))
    summary = document(run / "summary.json")
    evidence = run / "evidence.jsonl"
    batch = {"title": title, "path": run.resolve(), "evidence_version": file_version(evidence.stat()),
             "summary": summary, "sources": sources}
    rows = {}
    for offset, line in batch_lines(evidence, "索引识别区域", report):
        row = json.loads(line)
        target = row["models"][row["target_alias"]]
        entry = {key: row[key] for key in ("case_id", "doc_id", "page_id", "region_id", "task", "image", "page_image",
                 "image_sha256", "split", "priority", "reasons", "audit_sample", "hard_case", "signals",
                 "target_operational_error", "selection") if key in row}
        entry["selection"] = stored_selection(row, summary.get("mining"))
        entry["_offset"] = offset
        entry["valid_target"] = bool(target.get("label"))
        judge = target.get("judge", {})
        entry["target_state"] = {"status": target.get("status"),
                                 "judge": {key: judge.get(key) for key in ("status", "prob_yes")}}
        rows[row["case_id"]] = entry
    queue = {json.loads(line)["case_id"] for _, line in batch_lines(run / "review.jsonl", "读取复核队列", report)}
    details = {}
    references_path = pages.with_name("references.jsonl")
    if evaluation or references_path.is_file():
        with evidence_stream(batch) as stream:
            if evaluation:
                for _, line in batch_lines(evaluation / "cases.jsonl", "读取评测结果", report):
                    item = json.loads(line)
                    if item["case_id"] in rows:
                        row = evidence_row(stream, rows[item["case_id"]])
                        item["selection_gt"] = reference_comparison(row, item["models"])
                        details[item["case_id"]] = item
            else:
                by_region = {region_key(row): row for row in rows.values()}
                for _, line in batch_lines(references_path, "核对 GT", report):
                    try:
                        ref = json.loads(line)
                        entry = by_region.get(region_key(ref))
                        reference = ref["reference"]
                        if not entry or ref.get("image_sha256") != entry["image_sha256"] or not isinstance(reference, str) or not reference.strip():
                            continue
                        row = evidence_row(stream, entry)
                        metrics = reference_metrics(row, reference)
                        details[row["case_id"]] = {"reference": reference, "models": metrics,
                                                   "selection_gt": reference_comparison(row, metrics)}
                    except (KeyError, TypeError, ValueError):
                        continue
    return {**batch, "rows": rows, "queue": queue,
            "details": details, "selection_gt": selection_summary([item["selection_gt"] for item in details.values()]),
            "report": document(evaluation / "report.json") if evaluation else None}


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
        result = classify(row["target_state"], row["signals"], policy)
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


class ImportPath(BaseModel):
    path: str


def create_app(run=None, evaluation=None, production=None, library=None, manager_port=None, *, host="127.0.0.1"):
    batches = {}
    if run:
        title = "OmniDocBench 基准评测" if evaluation else run.name
        batches["benchmark"] = read_batch(run, evaluation, title)
    if production:
        batches["production"] = read_batch(production, None, "Paddle 原页采集")
    imports = {}
    import_lock = Lock()
    readers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="studio-import")

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            readers.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
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
        with evidence_stream(batch) as stream:
            return batch, evidence_row(stream, batch["rows"][case_id])

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html", headers={"Cache-Control": "no-cache"})

    def library_run(name):
        if not library or not name or name in {".", ".."} or Path(name).name != name:
            return None
        path = library / name / "production"
        if path.resolve().is_relative_to(library.resolve()) and all((path / file).is_file() for file in BATCH_FILES):
            return path
        return None

    def available_runs():
        if not library or not library.is_dir():
            return {}
        result = {}
        for path in sorted(library.iterdir(), reverse=True):
            run_path = library_run(path.name)
            if run_path:
                result[path.name] = run_path
        return result

    @app.get("/api/imports")
    def list_imports():
        runs = available_runs()
        items = []
        for name, path in runs.items():
            item = {"name": name}
            try:
                item["regions"] = document(path / "summary.json").get("scanned")
                item["first_doc"], cover = import_cover(path)
                item["has_cover"] = cover is not None
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
            items.append(item)
        return {"directory": str(library or ""), "names": list(runs), "batches": items}

    @app.get("/api/imports/{name}/cover")
    def batch_cover(name: str):
        run = library_run(name)
        if run:
            try:
                _, cover = import_cover(run)
                if cover:
                    return Response(page_thumbnail(cover, cover.stat().st_mtime_ns), media_type="image/jpeg",
                                    headers={"Cache-Control": "private, max-age=3600"})
            except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError):
                pass
        raise HTTPException(404, "批次没有可用的首页预览")

    def read_import(key, path, name, version):
        def progress(stage, filename, records, done, total):
            with import_lock:
                imports[key]["progress"] = {"state": "reading", "stage": stage, "file": filename,
                                            "records": records, "done": done, "total": total}

        try:
            batch = read_batch(path, None, name, progress)
            if import_version(path) != version:
                raise ValueError("结果文件在读取期间已更新，请重新导入")
        except Exception as error:
            with import_lock:
                imports[key]["progress"] = {"state": "error", "detail": f"批次读取失败：{error}"}
            return
        with import_lock:
            batches[key] = batch
            imports[key]["progress"] = {"state": "ready"}

    def open_import(path, name, background):
        path = path.resolve()
        identity = str(path)
        if library and path == library.resolve() / name / "production":
            identity = name  # Keep existing library batch IDs and their browser undo history.
        key = "import-" + digest(identity.encode())[:20]
        version = import_version(path)
        with import_lock:
            job = imports.get(key)
            if job is None or job["progress"]["state"] == "error" or (
                job["progress"]["state"] == "ready" and job["version"] != version
            ):
                job = {"version": version, "progress": {"state": "queued", "stage": "等待读取"}}
                imports[key] = job
                job["future"] = readers.submit(read_import, key, path, name, version)
            if background:
                return {"id": key, "name": name, **job["progress"]}
        job["future"].result()
        if job["progress"]["state"] == "error":
            raise HTTPException(422, job["progress"]["detail"])
        return {"id": key, "name": name}

    @app.get("/api/import-progress/{key}")
    def import_progress(key: str):
        with import_lock:
            if key not in imports:
                raise HTTPException(404, "导入任务不存在，请重新导入")
            return imports[key]["progress"]

    def resolve_import_path(source):
        raw = source.path.strip()
        if not raw or "\x00" in raw:
            raise HTTPException(422, "请输入服务器上的结果目录路径")
        try:
            directory = ((library or Path.cwd()) / Path(raw).expanduser()).resolve()
            if not directory.is_dir():
                raise HTTPException(422, f"目录不存在或无法访问：{directory}")
            path = directory / "production" if (directory / "production").is_dir() else directory
            missing = [name for name in BATCH_FILES if not (path / name).is_file()]
        except (OSError, RuntimeError) as error:
            raise HTTPException(422, f"无法打开目录：{error}") from error
        if missing:
            raise HTTPException(422, f"请指定已完成的批次或 production 目录。{path} 缺少 {', '.join(missing)}")
        return path

    @app.post("/api/import-preview")
    def preview_import(source: ImportPath):
        path = resolve_import_path(source)
        try:
            summary = document(path / "summary.json")
            return {"path": str(path), "regions": summary.get("scanned"), "review": summary.get("review"),
                    "hard_cases": summary.get("hard_cases"), "bytes": (path / "evidence.jsonl").stat().st_size}
        except (OSError, ValueError, AttributeError) as error:
            raise HTTPException(422, f"无法读取批次摘要：{error}") from error

    @app.post("/api/imports")
    def import_by_path(source: ImportPath, background: bool = False):
        path = resolve_import_path(source)
        return open_import(path, path.parent.name if path.name == "production" else path.name, background)

    @app.post("/api/imports/{name}")
    def import_batch(name: str, background: bool = False):
        path = library_run(name)
        if path is None:
            raise HTTPException(404, "批次不存在或仍在生成")
        return open_import(path, name, background)

    @app.get("/api/batches")
    def batch_list():
        with import_lock:
            return {key: batch["title"] for key, batch in batches.items()}

    def queue_entries(batch, recorded):
        for case_id, row in batch["rows"].items():
            target = row["target_state"]
            detail = batch["details"].get(case_id, {})
            comparison = detail.get("selection_gt", {})
            yield {
                **{key: row.get(key) for key in
                   ("case_id", "doc_id", "page_id", "region_id", "task", "priority", "reasons", "audit_sample", "hard_case")},
                "operational": bool(row.get("target_operational_error")),
                "valid_target": row["valid_target"],
                "automatic_model": row["selection"]["model_alias"],
                "selection_status": row["selection"]["status"],
                "selection_flagged": bool(row["selection"].get("reasons")),
                "queued": case_id in batch["queue"], "judge_status": target.get("judge", {}).get("status"),
                "review_status": recorded.get(case_id, "pending"),
                "target_error": detail.get("target_error"), "repairable": detail.get("repairable"),
                "gt_closest": comparison.get("is_closest") if comparison.get("valid_candidates", 0) >= 2 else None,
                "exportable": row.get("split", "mine") in SFT_SPLITS,
                "split": row.get("split", "mine"),
            }

    @app.get("/api/batches/{batch_id}")
    def batch_data(batch_id: str, stream: bool = False):
        original = original_batch(batch_id)
        with closing(decision_db(original["path"] / "review.sqlite")) as db:
            batch = active_policy(original, db)
            recorded = dict(db.execute("SELECT case_id, json_extract(payload, '$.status') FROM decisions"))
        metadata = {"summary": batch["summary"], "report": batch["report"], "sources": batch["sources"],
                    "selection_gt": batch["selection_gt"], "original_thresholds": threshold_values(original["summary"]["mining"])}
        rows = queue_entries(batch, recorded)
        if not stream:
            return JSONResponse({**metadata, "rows": list(rows)})

        def chunks():
            yield json.dumps({**metadata, "total": len(batch["rows"])}, ensure_ascii=False) + "\n"
            while group := list(islice(rows, 256)):
                yield json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n"

        return StreamingResponse(chunks(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/batches/{batch_id}/bulk")
    def bulk_pass(batch_id: str, ids: list[str]):
        batch = original_batch(batch_id)
        token = uuid4().hex
        changed = []
        with evidence_stream(batch) as stream, closing(decision_db(batch["path"] / "review.sqlite")) as db, db:
            db.execute("BEGIN IMMEDIATE")
            batch = active_policy(batch, db)
            for key in dict.fromkeys(ids):
                row = batch["rows"].get(key)
                if not row or row.get("hard_case") is not False or row.get("target_operational_error"):
                    continue
                if db.execute("SELECT 1 FROM decisions WHERE case_id=?", (key,)).fetchone():
                    continue
                if row["selection"]["status"] != "selected" or row["selection"].get("reasons"):
                    continue
                try:
                    record = decision_record(evidence_row(stream, row), "approved", row["selection"]["model_alias"])
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
            decision = db.execute("SELECT payload FROM decisions WHERE case_id = ?", (case_id,)).fetchone()
            saved_location = db.execute("SELECT payload FROM review_locations WHERE case_id = ?", (case_id,)).fetchone()
        manual_box = json.loads(saved_location[0])["box"] if saved_location else None
        return {
            **row, "metrics": detail.get("models", {}), "reference": reference, "selection_gt": detail.get("selection_gt"),
            "reference_view": output_view(reference, None, row["task"]), "views": views,
            "baseline_views": baseline_views,
            "has_crop": Path(row["image"]).is_file(),
            "page_geometry": page_geometry(row, manual_box) if has_page else None,
            "export_split": SFT_SPLITS.get(row.get("split", "mine")),
            "decision": json.loads(decision[0]) if decision else None,
            "policy": batch["summary"]["mining"],
        }

    @app.get("/api/batches/{batch_id}/cases/{case_id}/image/{kind}")
    def case_image(batch_id: str, case_id: str, kind: str):
        row = original_batch(batch_id)["rows"].get(case_id, {})
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

    @app.get("/api/batches/{batch_id}/export/selected")
    def export_selected(batch_id: str, final: bool = False):
        batch = original_batch(batch_id)
        overrides = decisions(batch["path"] / "review.sqlite") if final else {}

        def records():
            with evidence_stream(batch) as stream:
                for entry in batch["rows"].values():
                    result = selected_result(evidence_row(stream, entry))
                    if final:
                        decision = overrides.get(entry["case_id"])
                        if decision:
                            if decision["status"] != "approved":
                                continue
                            result.update(label=decision["label"], model_alias=decision["model_alias"],
                                          source=decision["source"], status="approved", margin=None,
                                          prob_yes=None, reasons=[])
                        elif result["model_alias"]:
                            result["source"] = "automatic"
                        else:
                            continue
                    yield json_text(result) + "\n"

        return StreamingResponse(records(),
                                 media_type="application/x-ndjson",
                                 headers={"Content-Disposition": "attachment; filename=" + ("labels.jsonl" if final else "selected.jsonl")})

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
            samples = sft_exports(batch, decisions(batch["path"] / "review.sqlite"))[split]
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if not samples:
            raise HTTPException(409, "该分区没有可导出的有效标签")
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
