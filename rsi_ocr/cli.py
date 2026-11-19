import argparse
from pathlib import Path

from .capture import capture_pages
from .config import load_config, resolve_config
from .mining import mine
from .reward import score, validate_weights
from .service import probe, serve
from .store import json_text
from .training import export_training, train


def absolute(text):
    return Path(text).resolve()


def main(argv=None):
    cli = argparse.ArgumentParser(prog="rsi-ocr")
    cli.add_argument("--config", default="configs/default.yaml", type=absolute)
    sub = cli.add_subparsers(dest="command", required=True)
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
    if args.command in {"capture", "mine", "export", "train"}:
        resolve_config(config, args.config.parent)
    if args.command == "capture":
        print(capture_pages(config, args.manifest, args.out))
    elif args.command == "mine":
        print(json_text(mine(config, args.manifest, args.out)))
    elif args.command == "export":
        print(json_text(export_training(config, args.evidence, args.review, args.out)))
    elif args.command == "train":
        return train(config, args.stage, args.data, args.out, args.checkpoint, args.dry_run)
    elif args.command == "reward":
        validate_weights(config["reward"])
        print(json_text(score(args.task, args.prediction.read_text(encoding="utf-8-sig"),
                              args.reference.read_text(encoding="utf-8-sig"), config["reward"])))
    return 0
