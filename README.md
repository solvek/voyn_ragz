# Volyn RAGZ

CLI-інструменти для обробки сканів карток РАГЗ (Волинська область, 1940-ті): розпізнавання рукописного тексту (TrOCR) і збереження сирого OCR у SQLite.

## Встановлення

З кореня репозиторію:

```bash
pip install -e .
```

На машині без GPU зручно спочатку поставити CPU-збірку PyTorch, потім пакет:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Після встановлення з’являться команди `recognize` і `train` (див. `pyproject.toml`). Альтернатива:

```bash
python -m volyn_ragz recognize --help
python -m volyn_ragz train --help
```

## Інструмент `recognize`

Ріже зображення на праву половину й горизонтальні смуги, проганяє їх через TrOCR і записує результат у таблицю `scan` (поле `raw_ocr`), якщо не вказано `--dry-run`.

**Синтаксис:** `recognize FOLDER [опції]` (аргумент `FOLDER` не потрібен разом із `--list-counties`).

`FOLDER` — підкаталог у корені сканів (за замовчуванням це `scans/FOLDER`). Імена файлів очікуються у вигляді `folder_file.jpeg` (до першого `_` — `folder`, після — `file` для БД).

**Коди району (`--county`).** Потрібен не довільний текст, а **код** з довідника в репозиторії:

- Файл **`volyn_ragz/counties.json`** — для кожного запису є `code` (його і передаєте в CLI) і `label` (повна українська назва; саме вона зберігається в `scan.county`).
- Команда **`recognize --list-counties`** (те саме через модуль: `python -m volyn_ragz recognize --list-counties`) виводить усі пари `код<TAB>назва` без завантаження моделі — зручно швидко підібрати потрібний код.

| Опція | Опис |
|--------|------|
| `--county` | **Обов’язково** для звичайного запуску. Код з `counties.json` (див. вище). |
| `--list-counties` | Вивести таблицю кодів і вийти; `FOLDER` не потрібен. |
| `--type` | Тип події в БД: `B` (народження), `M` (одруження), `D` (смерть), `R` (розлучення), `A` (усиновлення). |
| `--scans-root` | Каталог зі сканами (типово: `scans`). |
| `--db` | Шлях до SQLite (типово: `data/volyn_ragz.db`). |
| `--model` | Ідентифікатор моделі на Hugging Face **або локальний шлях** до збереженої дофайнтюненої моделі (див. нижче). Типово: `microsoft/trocr-base-handwritten`. |
| `--device` | `cpu` або `cuda`; якщо не вказано — автоматично (CUDA, якщо доступна). |
| `--skip-start N` | Пропустити перші N файлів після сортування (шум на початку каталогу). |
| `--skip-end N` | Пропустити останні N файлів. |
| `--limit N` | Обробити не більше N файлів (тести). |
| `--dry-run` | Лише вивід у stdout, без запису в БД. |

Повний перелік: `recognize --help`.

**Приклади:**

```bash
recognize --list-counties
recognize 122484190 --county kovelskyi_raion --type B
```

## Інструмент `train`

Дофайнтюнінг TrOCR на парах «зображення рядка — текст» (JSONL).

| Опція | Опис |
|--------|------|
| `--manifest` | **Обов’язково.** Файл JSONL: кожен рядок — об’єкт з ключами `image` (шлях до картинки) і `text` (еталонний текст рядка). |
| `--output-dir` | **Обов’язково.** Каталог, куди збережеться навчена модель (формат Hugging Face / Transformers). |
| `--base-model` | Базова модель (типово: `microsoft/trocr-base-handwritten`). |
| `--epochs` | Кількість епох навчання (типово: `3`). |
| `--batch-size` | Розмір батчу на пристрій (типово: `4`). |
| `--learning-rate` | Швидкість навчання (типово: `5e-5`). |

Повний перелік: `train --help`.

**Приклад:**

```bash
train --manifest data/lines.jsonl --output-dir models/volyn-trocr-v1
```

## Як підключити дофайнтюнену модель до `recognize`

Після `train` у `--output-dir` лежать `config.json`, ваги та файли процесора — це стандартний знімок для `from_pretrained`.

1. **Локально** — передайте шлях до цього каталогу в `--model`:

   ```bash
   recognize MY_FOLDER --county kovelskyi_raion --model ./models/volyn-trocr-v1
   ```

   Допустимі абсолютні та відносні шляхи; головне, щоб у каталозі були файли збереженої моделі.

2. **Hugging Face Hub** — якщо ви завантажили той самий каталог у репозиторій на Hub, використовуйте ідентифікатор репозиторію:

   ```bash
   recognize MY_FOLDER --county kovelskyi_raion --model your-username/your-volyn-trocr
   ```

   Перший запуск завантажить ваги в локальний кеш; далі можна працювати офлайн, якщо кеш уже є.

Базова модель з Hub без дофайнтюну залишається варіантом за замовчуванням (`--model microsoft/trocr-base-handwritten`), якщо `--model` не змінювати.
