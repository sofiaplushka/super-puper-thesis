from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml


def normalize_tag(tag: str | None) -> str:
    return " ".join(str(tag or "").strip().casefold().split())


def load_macro_mapping(path: str | Path) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): [str(v) for v in (vals or [])] for k, vals in data.items()}


def macro_lookup(mapping: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for macro, tags in mapping.items():
        for tag in tags:
            lookup[normalize_tag(tag)] = macro
    return lookup


def map_tags(tags: Iterable[str], mapping: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    lookup = macro_lookup(mapping)
    macros: list[str] = []
    unmapped: list[str] = []
    for tag in tags:
        norm = normalize_tag(tag)
        if not norm:
            continue
        macro = lookup.get(norm)
        if macro:
            macros.append(macro)
        else:
            unmapped.append(tag)
    ordered_macros = [m for m in mapping.keys() if m in set(macros)]
    if not ordered_macros:
        ordered_macros = ["other"]
    return ordered_macros, unmapped


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(raw, list):
        return [str(v) for v in raw]
    return [str(raw)]

