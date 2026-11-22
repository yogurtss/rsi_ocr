import argparse
from pathlib import Path

from .capture import capture_pages
from .config import load_config, resolve_config
from .mining import mine, warm
from .reward import score, validate_weights
from .service import probe, serve
from .store import json_text
from .training import export_training, train


def absolute(text):
    return Path(text).resolve()


def positive(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def fraction(text):
    number = float(text)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return number


def main(argv=None):
    cli = argparse.ArgumentParser(prog="rsi-ocr")
    cli.add_argument("--config", default="configs/default.yaml", type=absolute)
    sub = cli.add_subparsers(dest="command", required=True)
    pdf = sub.add_parser("run", help="process PDF/images in a file or directory and save a review batch")
    pdf.add_argument("input", type=absolute)
    pdf.add_argument("--settings", type=absolute, default=absolute("configs/import.yaml"))
    pdf.add_argument("--out", type=absolute, help="batch directory (default: timestamped)")
    pdf.add_argument("--import-only", action="store_true", help="render pages without starting models")
    start = sub.add_parser("serve", help="launch one model service from YAML")
    start.add_argument("alias")
    start.add_argument("--sidecar")
    start.add_argument("--show", action="store_true")
    check = sub.add_parser("probe", help="check model IDs and Jev readiness")
    check.add_argument("aliases", nargs="*")
    for name in ("capture", "mine"):
        run = sub.add_parser(name)
        run.add_argument("manifest", type=absolute)
        run.add_argument("--out", type=absolute, required=True)
        if name == "mine":
            run.add_argument("--cached-readers", action="store_true", help="score previously cached OCR responses")
    cache = sub.add_parser("warm", help="cache responses from one OCR service")
    cache.add_argument("alias")
    cache.add_argument("manifest", type=absolute)
    cache.add_argument("--out", type=absolute, required=True)
    prepare = sub.add_parser("prepare-omni", help="prepare OmniDocBench images and region manifests")
    prepare.add_argument("dataset", type=absolute)
    prepare.add_argument("--out", type=absolute, required=True)
    prepare.add_argument("--pages", type=positive, default=40)
    prepare.add_argument("--table-pages", type=positive, help="reserve this many pages with table annotations")
    prepare.add_argument("--regions-per-task", type=positive, help="sample at most this many regions per OCR task")
    prepare.add_argument("--download", action="store_true")
    evaluate = sub.add_parser("evaluate", help="evaluate mining against benchmark references")
    evaluate.add_argument("evidence", type=absolute)
    evaluate.add_argument("references", type=absolute)
    evaluate.add_argument("--out", type=absolute, required=True)
    evaluate.add_argument("--budget", type=positive, default=25)
    evaluate.add_argument("--partition", choices=("calibration", "holdout"), default="holdout")
    evaluate.add_argument("--text-cer", type=fraction, default=0.02, help="maximum CER for a correct text result")
    evaluate.add_argument("--table-teds", type=fraction, default=0.95, help="minimum TEDS for a correct table")
    export = sub.add_parser("export", help="export approved labels to ERNIEKit and ms-swift")
    export.add_argument("evidence", type=absolute)
    export.add_argument("review", type=absolute)
    export.add_argument("--out", type=absolute, required=True)
    fit = sub.add_parser("train", help="launch ERNIEKit SFT or ms-swift GRPO")
    fit.add_argument("stage", choices=("sft", "grpo"))
    fit.add_argument("--data", type=absolute, required=True)
    fit.add_argument("--out", type=absolute, required=True)
    fit.add_argument("--checkpoint")
    fit.add_argument("--dry-run", action="store_true")
    reward = sub.add_parser("reward", help="score one completion against a reference")
    reward.add_argument("task", choices=("text", "table"))
    reward.add_argument("prediction", type=absolute)
    reward.add_argument("reference", type=absolute)
    args = cli.parse_args(argv)
    if args.command == "run":
        from .importer import process_inputs

        try:
            process_inputs(args.input, args.settings, args.config, args.out, args.import_only)
        except (ValueError, FileNotFoundError) as error:
            cli.exit(1, f"{error}\n")
        return 0
    if args.command in {"prepare-omni", "evaluate"}:
        from . import benchmark

        if args.command == "prepare-omni":
            result = benchmark.prepare_omni(args.dataset, args.out, args.pages, args.download,
                                           table_pages=args.table_pages or 0, regions_per_task=args.regions_per_task)
        else:
            result = benchmark.evaluate(args.evidence, args.references, args.out, args.budget, args.partition,
                                        args.text_cer, args.table_teds)
        print(json_text(result))
        return 0
    config = load_config(args.config)
    if args.command == "serve":
        return serve(config, args.alias, args.sidecar, args.show)
    if args.command == "probe":
        policy = config["mining"]
        aliases = [policy["target"], *policy["teachers"]]
        if policy.get("judge"):
            aliases.append(policy["judge"])
        if policy.get("strong_teacher") and policy.get("strong_budget"):
            aliases.append(policy["strong_teacher"])
        return probe(config["models"], args.aliases or aliases)
    if args.command in {"capture", "mine", "warm", "export", "train"}:
        resolve_config(config, args.config.parent)
    if args.command == "capture":
        print(capture_pages(config, args.manifest, args.out))
    elif args.command == "mine":
        print(json_text(mine(config, args.manifest, args.out, args.cached_readers)))
    elif args.command == "warm":
        counts = warm(config, args.manifest, args.out, args.alias)
        print(json_text(counts))
        return 1 if counts.get("service_error") else 0
    elif args.command == "export":
        print(json_text(export_training(config, args.evidence, args.review, args.out)))
    elif args.command == "train":
        return train(config, args.stage, args.data, args.out, args.checkpoint, args.dry_run)
    elif args.command == "reward":
        validate_weights(config["reward"])
        print(json_text(score(args.task, args.prediction.read_text(encoding="utf-8-sig"),
                              args.reference.read_text(encoding="utf-8-sig"), config["reward"])))
    return 0
