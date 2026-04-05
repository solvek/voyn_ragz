"""CLI entry points: `recognize`, `train`."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

from volyn_ragz.counties import county_codes, county_label, iter_counties
from volyn_ragz.db import DEFAULT_DB_PATH, connect, upsert_scan_raw_ocr

# Як volyn_ragz.ocr.trocr_engine.DEFAULT_MODEL — без імпорту torch/transformers на старті.
DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"


def parse_scan_filename(name: str) -> tuple[str, str]:
    stem = Path(name).name
    if "_" not in stem:
        raise click.BadParameter(f"очікується ім'я виду folder_file.jpeg, отримано: {stem}")
    folder, rest = stem.split("_", 1)
    return folder, rest


IMAGE_RE = re.compile(r"\.(jpe?g|png|tif{1,2}|webp)$", re.I)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "folder",
    type=str,
    metavar="FOLDER",
    required=False,
)
@click.option(
    "--list-counties",
    is_flag=True,
    help="Вивести коди довідника районів і вийти (FOLDER не потрібен).",
)
@click.option(
    "--county",
    "county_code",
    required=False,
    type=click.Choice(county_codes(), case_sensitive=True),
    help="Код району з volyn_ragz/counties.json (зберігається в scan.county повна назва).",
)
@click.option(
    "--type",
    "event_type",
    type=click.Choice(["B", "M", "D", "R", "A"], case_sensitive=True),
    default=None,
    help="Тип події в БД: B народження, M одруження, D смерть, R розлучення, A усиновлення.",
)
@click.option(
    "--scans-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("scans"),
    help="Корінь каталогу зі сканами (типово: scans).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_DB_PATH,
    help="Шлях до SQLite БД.",
)
@click.option(
    "--model",
    default=DEFAULT_TROCR_MODEL,
    show_default=True,
    help="Ідентифікатор моделі TrOCR на Hugging Face.",
)
@click.option(
    "--device",
    default=None,
    help="cpu або cuda (за замовчуванням — автоматично).",
)
@click.option(
    "--skip-start",
    type=int,
    default=0,
    help="Пропустити перші N файлів після сортування.",
)
@click.option(
    "--skip-end",
    type=int,
    default=0,
    help="Пропустити останні N файлів після сортування.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Обробити не більше N файлів (для тестів).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Лише вивести рядки в stdout, без запису в БД.",
)
def recognize_main(
    folder: str | None,
    list_counties: bool,
    county_code: str | None,
    event_type: str | None,
    scans_root: Path,
    db_path: Path,
    model: str,
    device: str | None,
    skip_start: int,
    skip_end: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Розпізнає праву половину зображень у scans/FOLDER і пише сирий OCR у БД."""
    if list_counties:
        for code, label in iter_counties():
            click.echo(f"{code}\t{label}")
        return

    if not folder:
        raise click.UsageError("Потрібен FOLDER (підкаталог у scans), якщо не вказано --list-counties.")
    if not county_code:
        raise click.UsageError("Потрібен --county CODE; див. --list-counties.")

    county = county_label(county_code)
    from PIL import Image

    from volyn_ragz.image_prep import horizontal_strips, is_mostly_blank, right_half
    from volyn_ragz.ocr.trocr_engine import TrOCREngine

    scan_dir = scans_root / folder
    if not scan_dir.is_dir():
        raise click.ClickException(f"немає каталогу: {scan_dir}")

    files = sorted(
        p for p in scan_dir.iterdir() if p.is_file() and IMAGE_RE.search(p.name)
    )
    if skip_start:
        files = files[skip_start:]
    if skip_end and skip_end > 0:
        files = files[:-skip_end]
    if limit is not None:
        files = files[:limit]

    if not files:
        raise click.ClickException("не знайдено зображень за заданими фільтрами")

    engine = TrOCREngine(model_name=model, device=device)

    conn = None if dry_run else connect(db_path)
    try:
        for path in files:
            f_folder, f_file = parse_scan_filename(path.name)

            img = Image.open(path)
            rh = right_half(img)
            lines: list[str] = []
            for _, strip in horizontal_strips(rh):
                if is_mostly_blank(strip):
                    continue
                try:
                    text = engine.recognize_pil(strip)
                except Exception as e:  # noqa: BLE001
                    text = f"<error: {e}>"
                if text:
                    lines.append(text)
            raw = "\n".join(lines)
            if dry_run:
                click.echo(f"=== {path.name} ===\n{raw}\n")
            else:
                assert conn is not None
                upsert_scan_raw_ocr(
                    conn,
                    folder=f_folder,
                    file=f_file,
                    county=county,
                    raw_ocr=raw,
                    event_type=event_type,
                )
                conn.commit()
    finally:
        if conn is not None:
            conn.close()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL: кожен рядок {\"image\": \"...\", \"text\": \"...\"}.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куди зберегти дофайнтюнену модель.",
)
@click.option(
    "--base-model",
    default=DEFAULT_TROCR_MODEL,
    show_default=True,
)
@click.option("--epochs", type=float, default=3.0, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--learning-rate", type=float, default=5e-5, show_default=True)
def train_main(
    manifest: Path,
    output_dir: Path,
    base_model: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Дофайнтюнінг TrOCR на вирізках рядків (image + текст)."""
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{manifest}:{line_no}: {e}") from e
            if "image" not in row or "text" not in row:
                raise click.ClickException(
                    f"{manifest}:{line_no}: потрібні ключі image та text"
                )
            rows.append({"image": str(row["image"]), "text": str(row["text"])})

    if not rows:
        raise click.ClickException("manifest порожній")

    from volyn_ragz.train_trocr import run_finetune

    run_finetune(
        rows,
        output_dir=output_dir,
        base_model=base_model,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
    )
    click.echo(f"готово: {output_dir}")


def main() -> None:
    """Dispatch for `python -m volyn_ragz.cli`."""
    if len(sys.argv) < 2:
        click.echo("Використання: recognize … | train …", err=True)
        sys.exit(1)
    cmd = sys.argv[1]
    args = [sys.argv[0]] + sys.argv[2:]
    if cmd == "recognize":
        sys.argv = args
        recognize_main.main(standalone_mode=True)
    elif cmd == "train":
        sys.argv = args
        train_main.main(standalone_mode=True)
    else:
        click.echo("Невідома команда. Очікується recognize або train.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
