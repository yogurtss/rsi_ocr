"""Write the single-GPU WSL configuration without changing shared defaults."""
import argparse
import json
from rsi_ocr.local import configure_models
from pathlib import Path

cli = argparse.ArgumentParser()
cli.add_argument("--models", type=Path, default=Path("/mnt/g/models"))
cli.add_argument("--minicpm", type=Path, help="optional external MiniCPM-OCR repository")
cli.add_argument("--out", type=Path, default=Path("configs/local.yaml"))
args = cli.parse_args()
config = json.loads(Path("configs/default.yaml").read_text())
configure_models(config, args.models.resolve(), args.minicpm)
with args.out.open("x", encoding="utf-8") as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
