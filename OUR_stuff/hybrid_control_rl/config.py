"""YAML config loading helpers for the hybrid control pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge where values from override take precedence."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional `extends` key recursively."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    parent = data.pop("extends", None)
    if parent is None:
        return data

    parent_path = (path.parent / parent).resolve()
    return deep_merge(load_yaml_config(parent_path), data)

