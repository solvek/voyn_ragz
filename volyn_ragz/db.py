"""SQLite storage with WAL."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_DB_PATH = Path("data/volyn_ragz.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS scan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder TEXT NOT NULL,
    file TEXT NOT NULL,
    county TEXT NOT NULL,
    date TEXT,
    type TEXT,
    city TEXT,
    update_time TEXT NOT NULL,
    raw_ocr TEXT,
    UNIQUE(folder, file)
);

CREATE TABLE IF NOT EXISTS birth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL REFERENCES scan(id) ON DELETE CASCADE,
    surname TEXT,
    name TEXT,
    gender TEXT,
    number INTEGER,
    alive INTEGER,
    fsurname TEXT,
    fname TEXT,
    fpname TEXT,
    fage INTEGER,
    fetnicity TEXT,
    focupation TEXT,
    msurname TEXT,
    mname TEXT,
    mpname TEXT,
    mage INTEGER,
    metnicity TEXT,
    mocupation TEXT
);

CREATE TABLE IF NOT EXISTS marriage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL REFERENCES scan(id) ON DELETE CASCADE,
    msurname TEXT,
    msurname_old TEXT,
    mname TEXT,
    mpname TEXT,
    mage INTEGER,
    metnicity TEXT,
    mocupation TEXT,
    mstate TEXT,
    mchildren INTEGER,
    morder INTEGER,
    fsurname TEXT,
    fsurname_old TEXT,
    fname TEXT,
    fpname TEXT,
    fage INTEGER,
    fetnicity TEXT,
    focupation TEXT,
    fstate TEXT,
    fchildren INTEGER,
    forder INTEGER
);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def transaction(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_scan_raw_ocr(
    conn: sqlite3.Connection,
    *,
    folder: str,
    file: str,
    county: str,
    raw_ocr: str,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO scan (folder, file, county, date, type, city, update_time, raw_ocr)
        VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
        ON CONFLICT(folder, file) DO UPDATE SET
            county = excluded.county,
            update_time = excluded.update_time,
            raw_ocr = excluded.raw_ocr
        """,
        (folder, file, county, now, raw_ocr),
    )
    row = conn.execute(
        "SELECT id FROM scan WHERE folder = ? AND file = ?",
        (folder, file),
    ).fetchone()
    assert row is not None
    return int(row["id"])
