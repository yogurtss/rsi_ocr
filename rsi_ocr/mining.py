from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from .formats import NUMBER, normalize, similarity, table_cells, table_html, table_metrics
from .service import Service
from .store import Cache, atomic_output, digest, json_text, read_rows, write_rows


def candidate(result, task):
    if result["status"] != "ok":
        return result
    raw = result["raw"]
    try:
        label = table_html(table_cells(raw, result["format"])) if task == "table" else raw.strip()
    except ValueError as error:
        return {**result, "status": "format_invalid", "detail": str(error)}
    if not label:
        return {**result, "status": "empty"}
    return {**result, "label": label}


def compare(task, target, proposal):
    if task == "table":
        structure, content, numbers = table_metrics(table_cells(target), table_cells(proposal))
        return 1 - (0.5 * structure + 0.5 * content), numbers < 1
    left, right = normalize(target), normalize(proposal)
    return 1 - similarity(left, right), NUMBER.findall(left) != NUMBER.findall(right)


def evidence(case, inputs, target_alias, policy):
    target = inputs[target_alias]
    valid_teachers = [row for alias, row in inputs.items() if alias != target_alias and "label" in row]
    scores = []
    numbers_conflict = False
    target_score = target.get("judge", {}).get("prob_yes")
    by_family = defaultdict(list)
    for row in valid_teachers:
        if "label" in target:
            delta, conflict = compare(case["task"], target["label"], row["label"])
            row["distance_to_target"] = round(delta, 6)
            row["numeric_conflict"] = conflict
            numbers_conflict |= conflict
            by_family[row["family"]].append((delta, row.get("judge", {}).get("prob_yes")))
        if (score := row.get("judge", {}).get("prob_yes")) is not None:
            scores.append(score)
    family_scores = []
    for comparisons in by_family.values():
        weighted = [(distance, score) for distance, score in comparisons if score is not None]
        weight = sum(score for _, score in weighted)
        mean = sum(distance * score for distance, score in weighted) / weight if weight else (
            sum(distance for distance, _ in comparisons) / len(comparisons)
        )
        family_scores.append(mean)
    disagreement = sum(family_scores) / len(family_scores) if family_scores else None
    gain = max(0, max(scores) - target_score) if scores and target_score is not None else None
    signals = {
        "target_risk": 1 - target_score if target_score is not None else None,
        "disagreement": disagreement, "judge_gain": gain, "numeric_conflict": float(numbers_conflict) if family_scores else None,
    }
    return classify(target, signals, policy)


FIXED_RULES = ("numeric_conflict", "empty", "format_invalid", "judge_precheck_reject")


def classify(target, signals, policy):
    """Apply the mining policy to measured OCR evidence, including historical runs."""
    malformed = target["status"] in {"empty", "format_invalid"}
    target_score = target.get("judge", {}).get("prob_yes")
    disagreement = signals["disagreement"]
    gain = signals["judge_gain"]
    weights = policy["weights"]
    known = [key for key, value in signals.items() if value is not None and weights[key] > 0]
    risk = sum(weights[key] * signals[key] for key in known) / sum(weights[key] for key in known) if known else None
    reasons = []
    if malformed:
        reasons.append(target["status"])
    if target_score is not None and target_score <= policy["low_judge"]:
        reasons.append("low_target_judge")
    if disagreement is not None and disagreement >= policy["disagreement"]:
        reasons.append("teacher_disagreement")
    if gain is not None and gain >= policy["judge_gain"]:
        reasons.append("better_judged_candidate")
    if signals["numeric_conflict"]:
        reasons.append("numeric_conflict")
    if target.get("judge", {}).get("status") == "precheck_reject":
        reasons.append("judge_precheck_reject")
    switches = policy.get("fixed_rules", {})
    reasons = [reason for reason in reasons if reason not in FIXED_RULES or switches.get(reason, True)]
    operational = target["status"] in {"service_error", "incomplete"}
    return {"signals": signals, "hard_case": bool(reasons) and not operational,
            "priority": 1.0 if malformed else risk, "reasons": reasons,
            "target_operational_error": operational}


def proposals(case, inputs, target_alias, policy):
    target = inputs[target_alias]
    alternatives = [row for alias, row in inputs.items() if alias != target_alias and "label" in row]
    result = []
    for label in dict.fromkeys(row["label"] for row in alternatives):
        if "label" in target and compare(case["task"], target["label"], label)[0] <= 1e-12:
            continue
        supporting = [row for row in alternatives if compare(case["task"], row["label"], label)[0] <= 1e-12]
        families = sorted({row["family"] for row in supporting})
        judge = next(row.get("judge", {}) for row in supporting if row["label"] == label)
        probability = judge.get("prob_yes")
        recommended = len(families) >= policy["proposal_families"] and probability is not None and probability >= policy["proposal_judge"]
        result.append({"label": label, "families": families, "judge": judge, "recommended": recommended})
    return sorted(result, key=lambda row: (row["recommended"], len(row["families"]),
                                           row["judge"].get("prob_yes") or 0), reverse=True)


class Miner:
    def __init__(self, config, cache, base):
        self.policy = config["mining"]
        self.models = config["models"]
        self.services = {alias: Service(spec, cache) for alias, spec in self.models.items()}
        self.base = base

    def run(self, case, previous=None):
        task = case["task"]
        path = (self.base / case["image"]).resolve()
        image = path.read_bytes()
        image_hash = digest(image)
        if case.get("image_sha256", image_hash) != image_hash:
            raise ValueError(f"image changed after capture: {path}")
        if not image.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
            raise ValueError(f"expected PNG or JPEG crop: {path}")
        case_id = digest(json_text([case["doc_id"], case["page_id"], case["region_id"], task, image_hash]).encode())
        target = self.policy["target"]
        readers = [target, *self.policy["teachers"]] if previous is None else [self.policy["strong_teacher"]]
        inputs = {} if previous is None else previous["models"]
        for alias in readers:
            spec = self.models[alias]
            if task not in spec["tasks"]:
                continue
            result = case.get("target") if alias == target else None
            if result is None:
                result = self.services[alias].read(task, image)
            inputs[alias] = candidate({"model": spec["model"], "revision": spec["revision"],
                                       **result, "family": spec["family"]}, task)
        judge = self.policy.get("judge")
        if judge:
            existing = {row["label"]: row["judge"] for row in inputs.values()
                        if "label" in row and "judge" in row}
            labels = list(dict.fromkeys(row["label"] for row in inputs.values()
                                        if "label" in row and row["label"] not in existing))
            if labels:
                existing.update(zip(labels, self.services[judge].score(task, image, labels), strict=True))
            for row in inputs.values():
                if "label" in row:
                    row["judge"] = existing[row["label"]]
        return {**{key: value for key, value in case.items() if key != "target"},
                "case_id": case_id, "image_sha256": image_hash, "image": str(path), "models": inputs,
                "target_alias": target, **evidence(case, inputs, target, self.policy),
                "proposals": proposals(case, inputs, target, self.policy)}


def select(rows, policy):
    quota = policy["review_per_task"]
    per_doc = policy["review_per_document"]
    counts, documents, seen = Counter(), Counter(), set()
    hard = sorted((row for row in rows if row["hard_case"] and not row["audit_sample"]),
                  key=lambda row: (row["task"], -(row["priority"] or 0), row["case_id"]))
    chosen = [row for row in rows if row["audit_sample"] and not row["target_operational_error"]]
    for row in hard:
        task, doc, image = row["task"], row["doc_id"], row["image_sha256"]
        if counts[task] < quota[task] and documents[doc, task] < per_doc and (task, image) not in seen:
            counts[task] += 1
            documents[doc, task] += 1
            seen.add((task, image))
            chosen.append(row)
    return chosen


def eligible_regions(config, manifest):
    policy = config["mining"]
    target = config["models"][policy["target"]]
    doc_splits = {}
    region_ids = set()
    for row in read_rows(manifest):
        split = row.setdefault("split", "mine")
        doc_id = row["doc_id"]
        if doc_splits.setdefault(doc_id, split) != split:
            raise ValueError(f"document appears in multiple splits: {doc_id}")
        if split in policy["splits"] and row["task"] in target["tasks"]:
            key = doc_id, row["page_id"], row["region_id"], row["task"]
            if key in region_ids:
                raise ValueError(f"duplicate region: {key}")
            region_ids.add(key)
            yield row


def warm(config, manifest, output, alias):
    policy = config["mining"]
    if alias not in [policy["target"], *policy["teachers"]]:
        raise ValueError(f"{alias} is not a configured OCR reader")
    output.mkdir(parents=True, exist_ok=True)
    service = Service(config["models"][alias], Cache(output / "cache.sqlite"))

    def read(row):
        image = (manifest.parent / row["image"]).read_bytes()
        if row.get("image_sha256") not in (None, digest(image)):
            raise ValueError(f"image changed after capture: {row['image']}")
        return service.read(row["task"], image)["status"]

    rows = (row for row in eligible_regions(config, manifest) if row["task"] in service.config["tasks"]
            and not (alias == policy["target"] and "target" in row))
    with ThreadPoolExecutor(max_workers=policy["workers"]) as pool:
        counts = Counter()
        while batch := list(islice(rows, policy["workers"] * 4)):
            counts.update(pool.map(read, batch))
    return dict(counts)


def mine(config, manifest, output, cached_readers=False):
    output.mkdir(parents=True, exist_ok=True)
    policy = config["mining"]
    miner = Miner(config, Cache(output / "cache.sqlite"), manifest.parent)
    if cached_readers:
        for alias in [policy["target"], *policy["teachers"]]:
            miner.services[alias].cache_only = True

    rows = []
    workers = policy["workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = eligible_regions(config, manifest)
        while batch := list(islice(jobs, workers * 4)):
            rows.extend(pool.map(miner.run, batch))
        for row in rows:
            row["audit_probability"] = policy["audit_rate"]
            key = digest(f'{policy["audit_seed"]}:{row["case_id"]}'.encode())
            row["audit_sample"] = int(key[:16], 16) / 2**64 < policy["audit_rate"]
        if policy.get("strong_teacher") and policy.get("strong_budget", 0):
            candidates = [row for row in select(rows, policy) if row["hard_case"] and not any(
                item["recommended"] for item in row["proposals"]
            ) and row["task"] in miner.models[policy["strong_teacher"]]["tasks"]]
            chosen = candidates[:policy["strong_budget"]]
            enriched = pool.map(lambda row: miner.run(row, row), chosen)
            updates = {row["case_id"]: row for row in enriched}
            rows = [updates.get(row["case_id"], row) for row in rows]
    chosen = select(rows, policy)
    write_rows(output / "evidence.jsonl", rows)
    write_rows(output / "review.jsonl", chosen)
    write_rows(output / "candidate_labels.jsonl", (
        {"case_id": row["case_id"], "task": row["task"], "image_sha256": row["image_sha256"],
         "status": "unverified", **proposal}
        for row in chosen for proposal in row["proposals"]
    ))
    summary = {"scanned": len(rows), "review": len(chosen), "hard_cases": sum(row["hard_case"] for row in rows),
               "target_operational_errors": sum(row["target_operational_error"] for row in rows),
               "task_counts": dict(Counter(row["task"] for row in rows)),
               "model_status": {alias: dict(Counter(row["models"][alias]["status"] for row in rows
                                                    if alias in row["models"])) for alias in miner.models},
               "judge_status": dict(Counter(model["judge"]["status"] for row in rows
                                            for model in row["models"].values() if "judge" in model)),
               "mining": policy,
               "services": {alias: {key: spec[key] for key in ("model", "revision")}
                            for alias, spec in miner.models.items()}}
    with atomic_output(output / "summary.json") as temporary:
        temporary.write_text(json_text(summary) + "\n", encoding="utf-8")
    return summary
