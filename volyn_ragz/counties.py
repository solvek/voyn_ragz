"""Довідник районів / територій: коди CLI → назви для scan.county."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_COUNTIES_PATH = Path(__file__).with_name("counties.json")


@lru_cache(maxsize=1)
def _rows() -> tuple[dict[str, str], ...]:
    raw = json.loads(_COUNTIES_PATH.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for row in raw:
        out.append({"code": str(row["code"]), "label": str(row["label"])})
    return tuple(out)


def county_codes() -> tuple[str, ...]:
    return tuple(r["code"] for r in _rows())


def county_label(code: str) -> str:
    c = code.strip()
    for r in _rows():
        if r["code"] == c:
            return r["label"]
    raise KeyError(c)


def iter_counties() -> tuple[tuple[str, str], ...]:
    return tuple((r["code"], r["label"]) for r in _rows())
