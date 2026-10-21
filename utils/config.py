"""YAML configuration loading, dotted overrides, and path resolution."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    path = project_path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def parse_scalar(text):
    return yaml.safe_load(text)


def apply_overrides(config, overrides):
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        dotted, raw = item.split("=", 1)
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parse_scalar(raw)
    return result


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    path.write_text(
        yaml.safe_dump(clean, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
