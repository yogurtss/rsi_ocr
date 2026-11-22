# Repository Guidelines

## Project Structure

GlyphStack’s model console lives in `pageproof.py` and `manager.py` under `rsi_ocr/`. The same package contains the region-level PaddleOCR-VL pipeline. `cli.py` is the command entry point, `capture.py` records recognition crops, `service.py` handles model requests, and `mining.py` ranks cases for review. `formats.py` and `reward.py` handle text and table comparison, while `training.py` exports approved labels and launches SFT or GRPO. `configs/default.yaml` contains the shared defaults. `tests/test_pipeline.py` embeds its sample PNG and table fixtures. Keep generated crops, manifests, and training outputs under `runs/`.

## Build, Test, and Run Locally

Run these commands from the repository root with Python 3.10 or newer.

- `python -m pip install -e '.[quality]'` installs GlyphStack and the legacy `rsi-ocr` command.
- `glyphstack console` starts the model manager using `configs/import.yaml`. Run `glyphstack run --port 8520` to submit a batch and `glyphstack studio --port 8520` to open the review service.
- `python -m rsi_ocr --help` lists the CLI commands.
- `cp configs/default.yaml configs/local.yaml` creates a local configuration. Set model endpoints there and use `api_key_env` to reference credentials in environment variables.
- `rsi-ocr --config configs/local.yaml probe` checks configured model IDs and Jev readiness when the services are running. See `README.md` for setup and pipeline examples.

## Coding Style and Naming

Use four-space indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Follow nearby code for imports and formatting. The repository has no configured formatter or linter. Preserve raw model responses and crop bytes when changing capture or mining, and update model `revision` values when weights or prompts change so cached results are refreshed.

## Testing

Run `python -m unittest discover -s tests -v`. Add regression cases as `test_*.py` files with `test_` methods on `unittest.TestCase` classes. Use temporary directories for outputs and mock model requests so tests run without GPUs or live services. The suite covers capture and resume, text and table rewards, mining verdicts, and approved-label exports. There is no configured coverage threshold.

## Commits and Pull Requests

Use short imperative commit subjects, following examples such as `Initialize PaddleOCR-VL RSI project`. In pull requests, describe the behavior change and tests run, link related issues when available, and include a small sanitized JSONL example when changing a manifest or export format.
