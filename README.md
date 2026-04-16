# Volyn RAGZ

CLI-інструменти для обробки сканів карток РАГЗ (Волинська область, 1940-ті): розпізнавання рукописного тексту (TrOCR) і збереження сирого OCR у SQLite.

## Встановлення і перший запуск моделі (детально)

Нижче наведено максимально практичний сценарій "з нуля": перевіряємо GPU, ставимо правильний стек, кешуємо модель і запускаємо перевірку.

### 1) Перевірка відеокарти та CUDA

```bash
lspci | egrep -i "vga|3d|display|nvidia|amd|intel"
```

Якщо встановлена NVIDIA-карта і драйвер:

```bash
nvidia-smi
```

Швидка перевірка з PyTorch:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
```

### 2) Рекомендований варіант інсталяції (через локальне `.venv`)

> Рекомендовано завжди ставити в окреме середовище проєкту.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### Варіант A: CPU (без NVIDIA/CUDA, або є AMD/Intel GPU)

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-build-isolation
```

#### Варіант B: NVIDIA CUDA

```bash
python -m pip install -e . --no-build-isolation
```

Потім перевірити:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
PY
```

### 3) Перевірка сумісності `transformers` для TrOCR

Проєкт використовує `TrOCRProcessor` і `VisionEncoderDecoderModel`.
Іноді в нових версіях `transformers` може ламатися експорт `TrOCRProcessor`.

Перевірка:

```bash
python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
print("has_TrOCRProcessor:", hasattr(transformers, "TrOCRProcessor"))
PY
```

Якщо `has_TrOCRProcessor: False`, зафіксуйте сумісну версію:

```bash
python -m pip install "transformers==4.46.3"
```

### 4) Встановлення (кешування) базової моделі TrOCR

Базова модель:

- `microsoft/trocr-base-handwritten`

Попередньо завантажити в локальний кеш Hugging Face:

```bash
python - <<'PY'
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
model_id = "microsoft/trocr-base-handwritten"
TrOCRProcessor.from_pretrained(model_id)
VisionEncoderDecoderModel.from_pretrained(model_id)
print("Model is cached and ready.")
PY
```

Після цього `recognize` зможе працювати офлайн, якщо кеш не видаляти.

### 4.1) Скільки місця займає модель і чи це разово

- Для однієї базової моделі `microsoft/trocr-base-handwritten` орієнтуйтесь приблизно на **1-2 GB** диску (ваги + процесор + службові файли + кеш).
- Завантаження зазвичай **разове для цієї машини/середовища**: після кешування модель використовується локально.
- Якщо очищати Hugging Face cache або створити нове середовище без доступу до старого кешу, модель потрібно завантажити знову.

### 4.2) Ручне локальне завантаження моделі (рекомендовано для офлайн)

Якщо інтернет до Hub нестабільний, краще один раз зберегти модель у вашу папку проєкту й далі працювати тільки з локальним шляхом.

1. Створіть каталог під моделі:

```bash
mkdir -p models
```

2. Встановіть утиліту Hub (разово у `.venv`):

```bash
python -m pip install "huggingface_hub>=0.25"
```

3. Скачайте повний snapshot моделі у локальну папку:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/trocr-base-handwritten",
    local_dir="models/trocr-base-handwritten",
    local_dir_use_symlinks=False,
)
print("Downloaded to models/trocr-base-handwritten")
PY
```

4. Перевірте, що в папці є ключові файли (`config.json`, `preprocessor_config.json`, tokenizer/processor файли, ваги):

```bash
ls models/trocr-base-handwritten
```

5. Запускайте `recognize` вже з локальною моделлю:

```bash
recognize 122484190 --county kovelskyi_raion --model ./models/trocr-base-handwritten
```

Після цього доступ до інтернету для інференсу не потрібен.

### 5) Перевірка CLI

```bash
python -m volyn_ragz recognize --help
python -m volyn_ragz train --help
```

Після встановлення з’являються команди `recognize` і `train` (див. `pyproject.toml`), але запуск через модуль (`python -m ...`) надійніший і не залежить від PATH.

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

## Як донавчати локально завантажену модель

Локальна базова модель може бути джерелом для `train` так само, як і Hub-модель.

```bash
train \
  --manifest data/lines.jsonl \
  --base-model ./models/trocr-base-handwritten \
  --output-dir ./models/volyn-trocr-v1
```

Потім нову донавчену версію можна:

- використовувати для розпізнавання:

```bash
recognize 122484190 --county kovelskyi_raion --model ./models/volyn-trocr-v1
```

- або взяти як базу для наступного етапу донавчання (`--base-model ./models/volyn-trocr-v1`).

## Діагностика типових проблем

### `ImportError: cannot import name 'TrOCRProcessor'`

Причина: несумісна версія `transformers`.

Рішення:

```bash
python -m pip install "transformers==4.46.3"
```

### `torch.cuda.is_available() == False`

- На машині без NVIDIA це нормально (працюємо на CPU).
- Для примусового CPU-режиму в `recognize`:

```bash
recognize MY_FOLDER --county kovelskyi_raion --device cpu
```

### Команди `recognize` / `train` не знайдені

Запускайте через модуль:

```bash
python -m volyn_ragz recognize --help
python -m volyn_ragz train --help
```

Або перевстановіть пакет у активованому `.venv`:

```bash
python -m pip install -e . --no-build-isolation
```

### Не вдається скачати модель з Hugging Face (Connection reset / DNS / timeout)

- Для стабільної роботи зробіть разове ручне завантаження моделі у локальну папку (розділ `4.2`).
- Після цього вказуйте `--model ./models/trocr-base-handwritten` (або вашу донавчену локальну папку).
- Якщо модель уже в локальній папці, `recognize` і `train` можуть працювати без доступу до Hub.
