import os
import subprocess
from pathlib import Path

from .config import expand
from .formats import table_cells, table_html, table_otsl
from .reward import validate_weights
from .store import digest, json_text, read_rows, write_rows


def sft_record(image, prompt, answer):
    return {"image_info": [{"matched_text_index": 0, "image_url": str(image)}],
            "text_info": [{"text": prompt, "tag": "mask"}, {"text": answer, "tag": "no_mask"}]}


def paddle_sft_record(image, task, label):
    if task == "table":
        return sft_record(image, "Table Recognition:", table_otsl(table_cells(label)))
    if task == "text":
        return sft_record(image, "OCR:", label)
    raise ValueError(f"unsupported PaddleOCR-VL task: {task}")


def export_training(config, evidence_path, review_path, output):
    evidence = {row["case_id"]: row for row in read_rows(evidence_path)}
    target = config["models"][config["mining"]["target"]]
    grouped = {"train": [], "val": []}
    seen = {}
    decisions = set()
    for review in read_rows(review_path):
        case_id = review["case_id"]
        if case_id in decisions:
            raise ValueError(f"duplicate review decision: {case_id}")
        decisions.add(case_id)
        if review["status"] != "approved":
            continue
        row = evidence[case_id]
        if row.get("split") == "benchmark":
            continue
        image = (evidence_path.parent / row["image"]).resolve()
        if review["image_sha256"] != row["image_sha256"] or digest(image.read_bytes()) != row["image_sha256"]:
            raise ValueError(f"review image differs from captured image: {case_id}")
        split = {"mine": "train", "train": "train", "val": "val"}[row.get("split", "mine")]
        task = row["task"]
        spec = target["tasks"][task]
        label = review["label"].strip()
        if not label:
            raise ValueError(f"approved label is empty: {case_id}")
        solution = table_html(table_cells(label)) if task == "table" else label
        answer = table_otsl(table_cells(solution)) if spec["format"] == "otsl" else solution
        content_key = task, row["image_sha256"]
        if content_key in seen:
            if seen[content_key] != (split, solution):
                raise ValueError(f"conflicting label or split for the same image: {case_id}")
            continue
        seen[content_key] = split, solution
        grouped[split].append((row, review, str(image), spec["prompt"], answer, solution))
    for split, rows in grouped.items():
        write_rows(output / f"sft_{split}.jsonl", (
            sft_record(image, prompt, answer)
            for _, _, image, prompt, answer, _ in rows
        ))
        write_rows(output / f"grpo_{split}.jsonl", (
            {"images": [image], "messages": [{"role": "user", "content": "<image>" + prompt}],
             "solution": solution, "ocr_task": row["task"]}
            for row, _, image, prompt, _, solution in rows
        ))
    write_rows(output / "training_manifest.jsonl", (
        {"case_id": row["case_id"], "image_sha256": row["image_sha256"], "image": image, "split": split,
         "doc_id": row["doc_id"], "task": row["task"], "source": review["source"], "label": solution}
        for split, rows in grouped.items() for row, review, image, _, _, solution in rows
    ))
    return {split: len(rows) for split, rows in grouped.items()}


def train_command(config, stage, data, output, checkpoint):
    spec = config["training"][stage]
    args = [expand(arg) for arg in spec["argv"]]
    cwd = Path(expand(spec["cwd"])).resolve() if "cwd" in spec else None
    env = {**os.environ, **{key: expand(value) for key, value in spec.get("env", {}).items()}}
    dataset = str(data / f"{stage}_train.jsonl")
    val = data / f"{stage}_val.jsonl"
    validation = str(val) if val.exists() and val.stat().st_size else None
    model = str(Path(checkpoint).resolve()) if checkpoint and Path(checkpoint).exists() else (
        checkpoint or expand(spec["model"])
    )
    if stage == "sft":
        args.extend([f"model_name_or_path={model}", f"train_dataset_path={dataset}", f"output_dir={output}"])
        if validation:
            args.append(f"eval_dataset_path={validation}")
    else:
        validate_weights(config["reward"])
        target = config["models"][config["mining"]["target"]]
        if config["reward"]["table_format"] != target["tasks"]["table"]["format"]:
            raise ValueError("GRPO table output format must match the target model")
        env["RSI_OCR_REWARD_WEIGHTS"] = json_text(config["reward"])
        args.extend(["--model", model, "--dataset", dataset, "--output_dir", str(output),
                     "--external_plugins", str(Path(__file__).with_name("swift_reward.py")),
                     "--reward_funcs", "rsi_ocr_region"])
        if validation:
            args.extend(["--val_dataset", validation])
    return args, env, cwd


def train(config, stage, data, output, checkpoint, dry_run):
    args, env, cwd = train_command(config, stage, data, output, checkpoint)
    if dry_run:
        print(json_text({"argv": args, "cwd": str(cwd) if cwd else None,
                         "reward": config["reward"] if stage == "grpo" else None}))
        return 0
    dataset = data / f"{stage}_train.jsonl"
    if dataset.stat().st_size == 0:
        raise ValueError(f"empty training dataset: {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.call(args, env=env, cwd=cwd)
