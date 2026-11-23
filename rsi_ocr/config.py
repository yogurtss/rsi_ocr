import json
import os
from pathlib import Path
from string import Template


def load_config(path):
    path = Path(path).resolve()
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(raw)


def expand(value):
    return Template(value).substitute(os.environ) if "$" in value else value


def resolve_config(config, directory):
    for model in config["models"].values():
        if not model.get("enabled", True):
            continue
        for spec in model.get("tasks", {}).values():
            if "prompt_file" in spec:
                path = Path(expand(spec["prompt_file"]))
                spec["prompt"] = (directory / path).read_text(encoding="utf-8").strip()
    return config
