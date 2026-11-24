"""Prepare and evaluate recognition crops from OmniDocBench."""
import json
from math import isclose
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from urllib.parse import quote
from urllib.request import urlopen

from .formats import NUMBER, edit_distance, normalize, table_cells, table_metrics
from .mining import choose_transcription
from .store import atomic_output, digest, json_text, read_rows, write_rows

OMNI_REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
OMNI_URL = f"https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/{OMNI_REVISION}"
OMNI_ANNOTATIONS_SHA256 = "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"
TEXT_CATEGORIES = {
    "text_block", "title", "figure_caption", "table_caption", "table_footnote",
    "header", "footer", "page_number", "page_footnote", "aside_text",
}


def fetch(url, path):
    from shutil import copyfileobj

    with atomic_output(path) as temporary, urlopen(url, timeout=90) as response:
        with temporary.open("wb") as stream:
            copyfileobj(response, stream)


def prepare_omni(dataset, output, limit, download=False, seed=42, table_pages=0, regions_per_task=None):
    from PIL import Image

    annotations = dataset / "OmniDocBench.json"
    if download and not annotations.exists():
        fetch(f"{OMNI_URL}/OmniDocBench.json", annotations)
    data = annotations.read_bytes()
    annotations_hash = digest(data)
    if download and annotations_hash != OMNI_ANNOTATIONS_SHA256:
        raise ValueError(f"annotation hash differs from pinned OmniDocBench revision: {annotations}")
    groups = defaultdict(list)
    table_groups = defaultdict(list)
    for page in json.loads(data):
        info = page["page_info"]
        attributes = info.get("page_attribute") or {}
        source = attributes.get("data_source", "unknown")
        groups[source].append(page)
        if any(region["category_type"] == "table" and region.get("html") and not region.get("ignore")
               for region in page["layout_dets"]):
            table_groups[source].append(page)
    rng = random.Random(seed)
    for records in [*groups.values(), *table_groups.values()]:
        rng.shuffle(records)
    selected = []
    filenames = set()
    for pool, target_size in [(table_groups, min(table_pages, limit)), (groups, limit)]:
        while len(selected) < target_size and pool:
            for group in sorted(pool):
                while pool[group]:
                    record = pool[group].pop()
                    filename = record["page_info"]["image_path"]
                    if filename not in filenames:
                        filenames.add(filename)
                        selected.append(record)
                        break
                if not pool[group]:
                    del pool[group]
                if len(selected) == target_size:
                    break

    pages, regions, references = [], [], []
    excluded = Counter()
    for page in selected:
        info = page["page_info"]
        filename = Path(info["image_path"]).name
        source = dataset / "images" / filename
        if download and not source.exists():
            fetch(f"{OMNI_URL}/images/{quote(filename)}", source)
        doc_id = re.sub(r"(?:_page_|-page-|_)\d+$", "", Path(filename).stem)
        page_id = info["page_no"]
        partition = "calibration" if int(digest(doc_id.encode())[:8], 16) % 4 == 0 else "holdout"
        metadata = {"doc_id": doc_id, "page_id": page_id, "split": "benchmark",
                    "eval_partition": partition, "page_image": str(source.resolve())}
        pages.append({**metadata, "image": metadata["page_image"]})
        with Image.open(source) as image:
            width, height = image.size
            if (info["width"], info["height"]) != image.size:
                raise ValueError(f"annotation and image dimensions differ: {filename}")
            for annotation in page["layout_dets"]:
                category = annotation["category_type"]
                task = "table" if category == "table" else "text"
                if task == "text" and category not in TEXT_CATEGORIES:
                    continue
                label = annotation.get("html" if task == "table" else "text")
                if not isinstance(label, str) or not label.strip() or annotation.get("ignore"):
                    continue
                if task == "table":
                    try:
                        table_cells(label, "html")
                    except ValueError:
                        excluded["unsupported_table_annotation"] += 1
                        continue
                polygon = annotation["poly"]
                xs, ys = polygon[::2], polygon[1::2]
                box = (max(0, int(min(xs))), max(0, int(min(ys))),
                       min(width, int(max(xs)) + 1), min(height, int(max(ys)) + 1))
                if box[0] >= box[2] or box[1] >= box[3]:
                    excluded["invalid_region_bounds"] += 1
                    continue
                region_id = str(annotation["anno_id"])
                relative = Path("crops") / f"{Path(filename).stem}-{region_id}.png"
                with atomic_output(output / relative) as temporary:
                    image.crop(box).convert("RGB").save(temporary, format="PNG")
                row = {**metadata, "region_id": region_id, "task": task, "image": str(relative),
                       "bbox": box, "origin": "omnidocbench_annotation", "category": category,
                       "image_sha256": digest((output / relative).read_bytes())}
                references.append({key: row[key] for key in ("doc_id", "page_id", "region_id", "task")} | {
                    "reference": label, "image_sha256": row["image_sha256"]})
                regions.append(row)

    if regions_per_task is not None:
        chosen = set()
        for task in ("text", "table"):
            by_document = defaultdict(list)
            for row in regions:
                if row["task"] == task:
                    by_document[row["doc_id"]].append(row)
            documents = list(by_document)
            rng.shuffle(documents)
            for items in by_document.values():
                rng.shuffle(items)
            count = 0
            while by_document and count < regions_per_task:
                for doc_id in documents:
                    if doc_id not in by_document:
                        continue
                    chosen.add(region_key(by_document[doc_id].pop()))
                    count += 1
                    if not by_document[doc_id]:
                        del by_document[doc_id]
                    if count == regions_per_task:
                        break
        regions = [row for row in regions if region_key(row) in chosen]
        references = [row for row in references if region_key(row) in chosen]
    write_rows(output / "pages.jsonl", pages)
    write_rows(output / "regions.jsonl", regions)
    write_rows(output / "references.jsonl", references)
    result = {"pages": len(pages), "regions": len(regions), "seed": seed, "table_pages_requested": table_pages,
              "regions_per_task_limit": regions_per_task,
              "tasks": dict(Counter(row["task"] for row in regions)), "excluded": dict(excluded),
              "annotations_sha256": annotations_hash,
              "dataset_revision": OMNI_REVISION if annotations_hash == OMNI_ANNOTATIONS_SHA256 else None}
    with atomic_output(output / "dataset.json") as temporary:
        temporary.write_text(json_text(result), encoding="utf-8")
    return result


def region_key(row):
    return tuple(row[key] for key in ("doc_id", "page_id", "region_id", "task"))


def teds(predicted, reference, structure_only=False):
    from apted import APTED, Config
    from apted.helpers import Tree

    class TableEdit(Config):
        def rename(self, left, right):
            if left.name != right.name:
                return 1
            if left.name == "td":
                if left.span != right.span:
                    return 1
                if not structure_only:
                    return edit_distance(left.text, right.text) / max(len(left.text), len(right.text), 1)
            return 0

    def make_tree(cells):
        rows = [Tree("tr") for _ in range(max(cell.row + cell.height for cell in cells))]
        for cell in cells:
            node = Tree("td")
            node.span = cell.height, cell.width
            node.text = normalize(cell.text)
            rows[cell.row].children.append(node)
        return Tree("table", *rows), len(rows) + len(cells)

    left, left_size = make_tree(predicted)
    right, right_size = make_tree(reference)
    return 1 - APTED(left, right, TableEdit()).compute_edit_distance() / max(left_size, right_size)


def model_metrics(task, label, reference, text_cer, table_teds):
    if task == "text":
        pred, truth = normalize(label or ""), normalize(reference)
        cer = edit_distance(pred, truth) / max(1, len(truth))
        pred_numbers, ref_numbers = NUMBER.findall(pred), NUMBER.findall(truth)
        numbers = pred_numbers == ref_numbers if pred_numbers or ref_numbers else None
        return {"cer": cer, "numeric_match": numbers, "correct": cer <= text_cer and numbers is not False}
    truth = table_cells(reference)
    if not label:
        has_numbers = any(NUMBER.search(cell.text) for cell in truth)
        return {"teds": 0, "teds_s": 0, "numeric_accuracy": 0 if has_numbers else None, "correct": False}
    pred = table_cells(label)
    score = teds(pred, truth)
    structure = teds(pred, truth, structure_only=True)
    _, _, numbers = table_metrics(pred, truth)
    has_numbers = any(NUMBER.search(cell.text) for cells in (pred, truth) for cell in cells)
    return {"teds": score, "teds_s": structure, "numeric_accuracy": numbers if has_numbers else None,
            "correct": score >= table_teds and structure == 1 and numbers == 1}


def reference_comparison(row, metrics):
    """Audit a previously selected transcription against all valid OCR outputs."""
    metric = "cer" if row["task"] == "text" else "teds"
    scores = {alias: values[metric] for alias, values in metrics.items()
              if alias in row["models"] and row["models"][alias].get("status") == "ok"
              and row["models"][alias].get("label") and values.get(metric) is not None}
    best = (min if metric == "cer" else max)(scores.values(), default=None)
    closest = [alias for alias, value in scores.items() if isclose(value, best, rel_tol=0, abs_tol=1e-12)]
    selected = row.get("selection", {}).get("model_alias")
    return {"metric": metric, "selected_alias": selected, "selected_value": scores.get(selected),
            "best_value": best, "closest_aliases": closest, "valid_candidates": len(scores),
            "is_closest": selected in closest if selected in scores else None}


def selection_summary(comparisons):
    comparable = [item for item in comparisons if item["is_closest"] is not None and item["valid_candidates"] >= 2]
    return {"with_reference": len(comparisons), "comparable": len(comparable),
            "closest": sum(item["is_closest"] for item in comparable)}


def reference_metrics(row, reference, text_cer=0.02, table_teds=0.95):
    by_label = {}
    result = {}
    for alias, model in row["models"].items():
        if model["status"] in {"service_error", "incomplete"}:
            continue
        label = model.get("label")
        if label not in by_label:
            by_label[label] = model_metrics(row["task"], label, reference, text_cer, table_teds)
        result[alias] = by_label[label]
    return result


def evaluate(evidence_path, reference_path, output, budget, partition, text_cer=0.02, table_teds=0.95):
    references = {region_key(row): row for row in read_rows(reference_path)}
    evaluated = []
    details = []
    summary_path = evidence_path.with_name("summary.json")
    policy = json.loads(summary_path.read_bytes())["mining"] if summary_path.is_file() else None
    documents = set()
    excluded_operations = 0
    unmatched = 0
    for row in read_rows(evidence_path):
        if partition != "all" and row.get("eval_partition") != partition:
            continue
        ref = references.get(region_key(row))
        if ref is None or not (ref.get("reference") or "").strip():
            unmatched += 1
            continue
        if ref["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"reference crop differs from evidence: {row['case_id']}")
        if "selection" not in row and policy:
            row["selection"] = choose_transcription(row["models"], row["target_alias"], policy)
        metrics = reference_metrics(row, ref["reference"], text_cer, table_teds)
        detail = {"case_id": row["case_id"], "task": row["task"], "reference": ref["reference"],
                  "models": metrics, "selection_gt": reference_comparison(row, metrics)}
        details.append(detail)
        if row["target_operational_error"]:
            excluded_operations += 1
            continue
        documents.add(row["doc_id"])
        target_error = not metrics[row["target_alias"]]["correct"]
        repairable = target_error and any(result["correct"] for alias, result in metrics.items()
                                          if alias != row["target_alias"])
        distances = [model["distance_to_target"] for model in row["models"].values()
                     if "distance_to_target" in model]
        detail.update(target_error=target_error, repairable=repairable, disagreement=max(distances, default=0),
                      priority=row["priority"], hard_case=row["hard_case"])
        evaluated.append(detail)

    if not details:
        raise ValueError(f"no matched, evaluable cases in partition {partition!r}")
    strategies = {}
    for task in ("all", "text", "table"):
        rows = [row for row in evaluated if task == "all" or row["task"] == task]
        count = min(budget, len(rows))
        ordered = {
            "random": sorted(rows, key=lambda row: digest(f"42:{row['case_id']}".encode())),
            "disagreement": sorted(rows, key=lambda row: (-row["disagreement"], row["case_id"])),
            "full": sorted(rows, key=lambda row: (not row["hard_case"], -(row["priority"] or 0), row["case_id"])),
        }
        metrics = {}
        for name, ranking in ordered.items():
            top = ranking[:count]
            errors = sum(row["target_error"] for row in top)
            repairable = sum(row["repairable"] for row in top)
            total_errors = sum(row["target_error"] for row in rows)
            metrics[name] = {"reviewed": count, "errors_found": errors, "repairable": repairable,
                             "precision": errors / count if count else None,
                             "error_recall": errors / total_errors if total_errors else None,
                             "repairable_share": repairable / errors if errors else None}
        strategies[task] = metrics
    model_results = defaultdict(list)
    for row in evaluated:
        for alias, metrics in row["models"].items():
            model_results[row["task"], alias].append(metrics)
    model_summary = []
    for (task, alias), rows in sorted(model_results.items()):
        result = {"task": task, "model": alias, "cases": len(rows)}
        for key in rows[0]:
            values = [float(row[key]) for row in rows if row[key] is not None]
            result[key] = mean(values) if values else None
            if key.startswith("numeric"):
                result["numeric_cases"] = len(values)
        model_summary.append(result)
    metadata_path = reference_path.with_name("dataset.json")
    metadata = json.loads(metadata_path.read_bytes()) if metadata_path.is_file() else None
    report = {"partition": partition, "cases": len(evaluated), "documents": len(documents), "budget": budget,
              "dataset": metadata, "excluded_target_operations": excluded_operations,
              "unmatched": unmatched, "text_cer_threshold": text_cer, "table_teds_threshold": table_teds,
              "model_metrics": model_summary, "strategies": strategies}
    report["selection_gt"] = selection_summary([item["selection_gt"] for item in details])
    write_rows(output / "cases.jsonl", details)
    with atomic_output(output / "report.json") as temporary:
        temporary.write_text(json_text(report), encoding="utf-8")
    return report
