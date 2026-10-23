# Відтворюване середовище експерименту ПР-6: Linux (Debian slim) + Python 3.10.
# Базовий образ зафіксовано за версією і дайджестом, щоб збірка не залежала від оновлень тегу.
FROM python:3.10.22-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

# Коміт коду передається під час збірки: docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) ...
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Встановлення системних утиліт
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Інсталяція Python-залежностей: прямі залежності + зафіксовані версії транзитивних (constraints.txt)
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Копіювання коду експерименту та еталонних метрик для перевірки відтворюваності
COPY prepare.py train.py ./
COPY results/reference_metrics.json results/

CMD ["python", "train.py"]
