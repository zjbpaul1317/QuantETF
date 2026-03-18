from __future__ import annotations

import ast
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback path depends on environment
    yaml = None

from .schema import AppConfig
from .validator import validate_app_config

DEFAULT_CONFIG_FILES = (
    "base.yaml",
    "universe.yaml",
    "strategy.yaml",
    "backtest.yaml",
    "live.yaml",
    "logging.yaml",
)


def _strip_comments(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)

    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_comments(raw_line.rstrip("\n"))
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            lines.append((indent, line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        current_indent, current_text = lines[index]
        if current_indent != indent:
            raise ValueError(f"Invalid indentation near line {index + 1} in {path}")

        if current_text.startswith("- "):
            values: list[Any] = []
            while index < len(lines):
                current_indent, current_text = lines[index]
                if current_indent < indent:
                    break
                if current_indent != indent or not current_text.startswith("- "):
                    raise ValueError(f"Invalid list syntax near line {index + 1} in {path}")

                item_text = current_text[2:].strip()
                index += 1
                if item_text:
                    values.append(_parse_scalar(item_text))
                    continue

                nested, index = parse_block(index, indent + 2)
                values.append(nested)
            return values, index

        values_dict: dict[str, Any] = {}
        while index < len(lines):
            current_indent, current_text = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"Invalid mapping indentation near line {index + 1} in {path}")
            if ":" not in current_text:
                raise ValueError(f"Invalid mapping entry near line {index + 1} in {path}")

            key, remainder = current_text.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            index += 1
            if remainder:
                values_dict[key] = _parse_scalar(remainder)
                continue

            if index >= len(lines) or lines[index][0] <= indent:
                values_dict[key] = {}
                continue

            nested, index = parse_block(index, indent + 2)
            values_dict[key] = nested
        return values_dict, index

    if not lines:
        return {}

    payload, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError(f"Failed to parse full YAML file: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"Config file must contain a mapping at root: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_env_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_env_overrides(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    merged = deepcopy(payload)
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        path = key[len(prefix):].lower().split("__")
        if not path:
            continue

        cursor = merged
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _parse_env_value(value)
    return merged


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if yaml is None:
        return _load_simple_yaml(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config file must contain a mapping at root: {config_path}")
    return payload


def load_raw_config(
    config_dir: str | Path = "configs",
    filenames: Iterable[str] = DEFAULT_CONFIG_FILES,
    env_prefix: str | None = "QUANT_ETF_",
) -> dict[str, Any]:
    directory = Path(config_dir).resolve()
    merged: dict[str, Any] = {}
    for filename in filenames:
        merged = _deep_merge(merged, load_yaml_file(directory / filename))

    if env_prefix:
        merged = _apply_env_overrides(merged, env_prefix)
    return merged


def load_app_config(
    config_dir: str | Path = "configs",
    filenames: Iterable[str] = DEFAULT_CONFIG_FILES,
    env_prefix: str | None = "QUANT_ETF_",
) -> AppConfig:
    directory = Path(config_dir).resolve()
    raw = load_raw_config(directory, filenames=filenames, env_prefix=env_prefix)
    config = AppConfig.from_dict(raw, project_root=directory.parent)
    validate_app_config(config)
    return config
