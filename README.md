# Відтворюваний експеримент: класифікація структур переплетення за текстурами DTD

Повний відтворюваний пакет обчислювального експерименту (Reproducible Research Artifact Bundle) до
практичної роботи 6 з курсу «Методологія та організація наукових досліджень» (ХНУ, 2026).
Частина магістерського дослідження «Автоматизоване визначення лінійної густини пряжі за цифровими
зображеннями тканини».

**Задача.** Класифікація 4 структур переплетення – `woven`, `knitted`, `braided`, `meshed` – за 104 ознаками
текстури (статистика яскравості, спектр Фур'є, LBP), обчисленими із зображень
[Describable Textures Dataset](https://doi.org/10.1109/CVPR.2014.461) (480 зображень, по 120 на клас).
Алгоритм – RandomForest у двох конфігураціях: `rf_baseline` (100 дерев, глибина 5) і `rf_full` (300 дерев).

**Що забезпечує відтворюваність**

| Рівень | Засіб |
|---|---|
| Код | Git, реліз `v1.0.0` |
| Програмні залежності | `requirements.txt` (прямі, точні версії) + `constraints.txt` (усі транзитивні) |
| Система й Python | `Dockerfile` на `python:3.10.22-slim`, зафіксованому за дайджестом |
| Дані | DVC: `data/dataset.csv.dvc` з MD5; `prepare.py` відтворює файл із першоджерела за зафіксованою ревізією |
| Випадковість | `random_state = 42` у розбитті й моделі, `PYTHONHASHSEED=0` |
| Параметри й метрики | MLflow Tracking: гіперпараметри, метрики, MD5 даних, коміт Git, середовище, модель |
| Перевірка | `results/reference_metrics.json` – еталонні метрики; `train.py` порівнює з ними кожен запуск |
| Метадані | `zenodo.json` / `.zenodo.json` (Zenodo, DOI), `CITATION.cff`, ліцензія MIT |

## Структура репозиторію

```
reproducible-experiment/
├── prepare.py                 # DTD (Hugging Face Hub, фікс. ревізія) -> data/dataset.csv
├── train.py                   # навчання й оцінка RandomForest, логування в MLflow
├── requirements.txt           # прямі залежності з точними версіями
├── constraints.txt            # повний перелік версій (pip freeze) для бітової точності середовища
├── Dockerfile                 # ізольоване середовище Linux + Python 3.10
├── docker-compose.yml         # сервер MLflow + контейнер експерименту
├── data/dataset.csv.dvc       # вказівник DVC на датасет (MD5, розмір)
├── results/reference_metrics.json
├── zenodo.json, .zenodo.json  # метадані для публікації на Zenodo
├── CITATION.cff, LICENSE
└── .github/workflows/reproduce.yml   # незалежне відтворення в GitHub Actions
```

## Відтворення в один клік (Docker Compose)

Потрібні лише Git і Docker (Docker Desktop у Windows/macOS або Docker Engine у Linux).

```bash
git clone https://github.com/Lepestochek1/reproducible-experiment.git
cd reproducible-experiment
docker compose up --build
```

Compose збирає образ `cs-experiment:v1.0`, піднімає сервер MLflow, чекає, доки він відповість, і запускає
експеримент. Результати – у вебінтерфейсі **http://localhost:5000** (експеримент `reproducibility-benchmark`,
запуски `rf_baseline` і `rf_full`). У кінці журналу контейнера `experiment` має з'явитися рядок
`ВІДТВОРЕНО: метрики збігаються з еталонними`. Зупинка сервера – `Ctrl+C`, видалення контейнерів –
`docker compose down`.

Якщо файлу `data/dataset.csv` немає, контейнер сам виконає `prepare.py` (потрібен інтернет для завантаження
480 зображень DTD, ≈ 60 МБ).

## Покрокове відтворення

### 1. Отримати дані (DVC)

```bash
pip install dvc
dvc pull                 # якщо налаштовано віддалене сховище DVC
# або відтворити датасет із першоджерела (Hugging Face Hub, ревізія 96e70d3…):
python prepare.py
dvc status               # "Data and pipelines are up to date" – файл збігся з версією в Git за MD5
```

### 2. Зібрати Docker-образ

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t cs-experiment:v1.0 .
```

### 3. Запустити експеримент у контейнері

Linux / macOS / Git Bash:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/mlruns:/app/mlruns" \
    -v "$(pwd)/results:/app/results" cs-experiment:v1.0
```

Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/mlruns:/app/mlruns" `
    -v "${PWD}/results:/app/results" cs-experiment:v1.0
```

Метрики запуску зберігаються в `results/metrics_docker.json`. Порівняння з еталоном виконується автоматично;
`python train.py --strict` завершується з кодом 1, якщо метрики не збіглися.

### 4. Переглянути логи в MLflow UI

```bash
mlflow ui --port 5000    # відкрити http://localhost:5000
```

Файлове сховище `mlruns/`, створене в контейнері, зберігає шляхи артефактів `/app/mlruns/...`, тому
параметри й метрики в UI на хост-машині видно, а артефакти моделей – лише через Docker Compose (сервер MLflow
зберігає артефакти в `mlflow-server/artifacts`).

### 5. Без Docker (локальне віртуальне середовище)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
python train.py
mlflow ui --port 5000
```

## Еталонні результати

Еталонні метрики на Validation (96 зображень, стратифіковане розбиття 80/20, `random_state = 42`) наведені
в `results/reference_metrics.json` (отримано на Windows 11, Python 3.10.2, датасет MD5
`35bda8820494391c51e8a008428a5b05`, коміт `520b33e`):

| Запуск | n_estimators | max_depth | F1 (weighted) | Accuracy | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| `rf_baseline` | 100 | 5 | 0.5814 | 0.5833 | 0.8449 |
| `rf_full` | 300 | – | 0.6435 | 0.6458 | 0.8662 |

Повторний запуск у тому самому середовищі дав ідентичні метрики (Repeatability за ACM).

## Публікація на Zenodo та присвоєння DOI

1. Увійти на <https://zenodo.org> через GitHub і в розділі *GitHub* увімкнути інтеграцію для цього репозиторію.
2. Створити на GitHub реліз із тегом `v1.0.0` – Zenodo автоматично архівує його, бере метадані з
   `.zenodo.json` і присвоює DOI.
3. Додати отриманий DOI-бейдж у цей README і поле `doi` у `CITATION.cff`.

Файли `zenodo.json` і `.zenodo.json` однакові: перший – назва з методичних рекомендацій, другий читає
інтеграція GitHub–Zenodo.

## Ліцензія та цитування

Код – ліцензія MIT (`LICENSE`). Датасет DTD належить його авторам і поширюється на їхніх умовах; у репозиторії
зберігаються лише обчислені ознаки (під контролем DVC). Як цитувати – `CITATION.cff`.

Cimpoi M., Maji S., Kokkinos I., Mohamed S., Vedaldi A. Describing Textures in the Wild. CVPR 2014.
DOI: 10.1109/CVPR.2014.461.
