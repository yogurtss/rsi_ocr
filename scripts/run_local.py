"""Run benchmark and optional production manifests through local model services."""
import argparse
from pathlib import Path

from rsi_ocr.config import load_config, resolve_config
from rsi_ocr.local import run_local


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    cli.add_argument("--regions", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--pages", type=Path, help="also capture the actual Paddle crops")
    cli.add_argument("--stage", choices=["all", "paddle", "oxr", "ovis", "jev"], default="all")
    args = cli.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    resolve_config(config, args.config.resolve().parent)
    jobs = [(args.regions.resolve(), output / "benchmark")]
    if args.pages:
        jobs.append((output / "capture" / "regions.jsonl", output / "production"))
    run_local(config, output, jobs, args.pages, args.stage)


if __name__ == "__main__":
    main()
