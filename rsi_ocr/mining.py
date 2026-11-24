from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def choose_transcription(models, target, policy):
    order = dict.fromkeys([target, *policy.get("teachers", []), *sorted(models)])
    valid = [key for key in order if key in models and models[key].get("status") == "ok" and models[key].get("label")]
    choice = {"model_alias": None, "prob_yes": None, "margin": None, "status": "unavailable", "reasons": []}
    if not valid:
        return {**choice, "reasons": ["no_valid_output"]}

    ranked = []
    partial = False
    for alias in valid:
        judge = models[alias].get("judge", {})
        score = judge.get("prob_yes")
        if policy.get("judge") and judge.get("status") == "scored" and isinstance(score, (float, int)) and 0 <= score <= 1:
            ranked.append((alias, score))
        elif judge.get("status") != "precheck_reject":
            partial = True
    if not ranked:
        enabled = bool(policy.get("judge"))
        pool = [alias for alias in valid if not enabled or
                models[alias].get("judge", {}).get("status") != "precheck_reject"]
        reasons = ["no_scored_output" if enabled else "judge_disabled"]
        if not pool:
            pool = valid
            reasons = ["all_precheck_rejected"]
        families = defaultdict(set)
        labels = {alias: normalize(models[alias]["label"]) for alias in pool}
        for alias, label in labels.items():
            families[label].add(models[alias].get("family") or alias)
        winner = max(pool, key=lambda alias: len(families[labels[alias]]))
        support = len(families[labels[winner]])
        tied = sum(len(group) == support for group in families.values()) > 1
        if support > 1:
            reasons.append("family_agreement")
        if tied:
            reasons.append("policy_tiebreak")
        elif len(pool) == 1:
            reasons.append("sole_candidate")
        return {**choice, "model_alias": winner, "status": "selected", "reasons": reasons}
    ranked.sort(key=lambda item: item[1], reverse=True)
    winner, score = ranked[0]
    second = next((value for alias, value in ranked if models[alias]["label"] != models[winner]["label"]), None)
    margin = round(score - second, 8) if second is not None else None
    reasons = []
    if score < policy["proposal_judge"]:
        reasons.append("low_score")
    if margin is not None and (margin == 0 or margin < policy.get("selection_margin", 0.05)):
        reasons.append("close_scores")
    if partial:
        reasons.append("partial_scores")
    return {"model_alias": winner, "prob_yes": score, "margin": margin,
            "status": "selected", "reasons": reasons}


def stored_selection(row, policy):
    selection = row.get("selection")
    if policy and (not selection or selection.get("reasons") in (["no_scored_output"], ["judge_disabled"])):
        return choose_transcription(row["models"], row["target_alias"], policy)
    if selection and selection.get("status") == "needs_check" and selection.get("model_alias"):
        return {**selection, "status": "selected"}
    return selection


def selected_result(row):
    selection = row["selection"]
    alias = selection["model_alias"]
    return {**{key: row[key] for key in ("case_id", "doc_id", "page_id", "region_id", "task", "image_sha256")},
            **selection, "label": row["models"][alias]["label"] if alias else None}


def automatic_label(row, selection):
    alias = selection and selection.get("model_alias")
    if alias:
        return {"case_id": row["case_id"], "image_sha256": row["image_sha256"],
                "source": "automatic", "model_alias": alias, "label": row["models"][alias]["label"]}


class Miner:
    def __init__(self, config, cache, base):
        self.policy = config["mining"]
        self.models = {alias: spec for alias, spec in config["models"].items() if spec.get("enabled", True)}
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
                existing.update(zip(labels, self.services[judge].score(task, image, labels, region=case), strict=True))
            for row in inputs.values():
                if "label" in row:
                    row["judge"] = existing[row["label"]]
        return {**{key: value for key, value in case.items() if key != "target"},
                "case_id": case_id, "image_sha256": image_hash, "image": str(path), "models": inputs,
                "target_alias": target, **evidence(case, inputs, target, self.policy),
                "selection": choose_transcription(inputs, target, self.policy),
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


def warm(config, manifest, output, alias, check_cancelled=None, resume=None, report_work=None, region_totals=None):
    policy = config["mining"]
    if alias not in [policy["target"], *policy["teachers"]]:
        raise ValueError(f"{alias} is not a configured OCR reader")
    output.mkdir(parents=True, exist_ok=True)
    service = Service(config["models"][alias], Cache(output / "cache.sqlite"), resume=resume)

    def read(row):
        if check_cancelled:
            check_cancelled()
        if report_work:
            report_work(alias, row["doc_id"], started=True)
        image = (manifest.parent / row["image"]).read_bytes()
        if row.get("image_sha256") not in (None, digest(image)):
            raise ValueError(f"image changed after capture: {row['image']}")
        status = service.read(row["task"], image, region=row)["status"]
        if report_work:
            report_work(alias, row["doc_id"], failed=status == "service_error")
        return status

    def work_items():
        final = policy.get("judge") or "mine"
        for row in eligible_regions(config, manifest):
            if region_totals is not None:
                if check_cancelled:
                    check_cancelled()
                for name, totals in region_totals.items():
                    if name == final or row["task"] in config["models"][name]["tasks"]:
                        totals[row["doc_id"]] += 1
            if row["task"] in service.config["tasks"] and not (alias == policy["target"] and "target" in row):
                yield row

    workers = max(policy["workers"], service.config.get("max_sequences", 0))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = Counter()
        pending = set()
        try:
            for row in work_items():
                # Keep one worker-pool's worth of regions ready while slower requests finish.
                if len(pending) >= 2 * workers:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    counts.update(future.result() for future in completed)
                pending.add(pool.submit(read, row))
            counts.update(future.result() for future in pending)
        finally:
            for future in pending:
                future.cancel()
    return dict(counts)


def mine(config, manifest, output, cached_readers=False, check_cancelled=None, resume=None, report_work=None):
    output.mkdir(parents=True, exist_ok=True)
    policy = config["mining"]
    miner = Miner(config, Cache(output / "cache.sqlite"), manifest.parent)
    if judge := policy.get("judge"):
        miner.services[judge].resume = resume
    if cached_readers:
        for alias in [policy["target"], *policy["teachers"]]:
            miner.services[alias].cache_only = True

    def score_region(row, initial=None):
        if check_cancelled:
            check_cancelled()
        if report_work and initial is None:
            report_work(policy.get("judge") or "mine", row["doc_id"], started=True)
        result = miner.run(row, initial)
        if report_work and initial is None:
            failed = any(model.get("judge", {}).get("status") in
                         {"service_error", "forward_error", "render_error", "encode_error"}
                         for model in result["models"].values())
            report_work(policy.get("judge") or "mine", row["doc_id"], failed=failed)
        return result

    rows = []
    workers = policy["workers"]
    if judge := policy.get("judge"):
        workers = max(workers, miner.models.get(judge, {}).get("max_sequences", 0))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = eligible_regions(config, manifest)
        while batch := list(islice(jobs, workers * 4)):
            rows.extend(pool.map(score_region, batch))
        for row in rows:
            row["audit_probability"] = policy["audit_rate"]
            key = digest(f'{policy["audit_seed"]}:{row["case_id"]}'.encode())
            row["audit_sample"] = int(key[:16], 16) / 2**64 < policy["audit_rate"]
        if policy.get("strong_teacher") and policy.get("strong_budget", 0):
            candidates = [row for row in select(rows, policy) if row["hard_case"] and not any(
                item["recommended"] for item in row["proposals"]
            ) and row["task"] in miner.models[policy["strong_teacher"]]["tasks"]]
            chosen = candidates[:policy["strong_budget"]]
            enriched = pool.map(lambda row: score_region(row, row), chosen)
            updates = {row["case_id"]: row for row in enriched}
            rows = [updates.get(row["case_id"], row) for row in rows]
    if check_cancelled:
        check_cancelled()
    chosen = select(rows, policy)
    write_rows(output / "evidence.jsonl", rows)
    write_rows(output / "selected.jsonl", (selected_result(row) for row in rows))
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
    from .training import batch_labels, write_training

    evidence = {row["case_id"]: row for row in rows}
    evidence_path = output / "evidence.jsonl"
    summary["selected"] = sum(bool(row["selection"]["model_alias"]) for row in rows)
    summary["training"] = write_training(config, evidence_path, evidence,
                                         batch_labels(evidence, evidence_path, policy), output / "training")
    with atomic_output(output / "summary.json") as temporary:
        temporary.write_text(json_text(summary) + "\n", encoding="utf-8")
    return summary
