"""Відтворюваний обчислювальний експеримент ПР-6 з трекінгом MLflow.

Класифікація структур переплетення (woven, knitted, braided, meshed) за ознаками текстур DTD
алгоритмом RandomForest. Кожен запуск реєструє в MLflow гіперпараметри, метрики якості, час навчання,
MD5 датасету, коміт Git, середовище виконання та артефакти (модель, матриця помилок, звіт класифікації).
Після запуску метрики порівнюються з еталонними (results/reference_metrics.json).

Запуск:  python train.py               # усі конфігурації
         python train.py --runs rf_baseline
Сервер MLflow задається змінною MLFLOW_TRACKING_URI, інакше логи пишуться в ./mlruns.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
EXPERIMENT = 'reproducibility-benchmark'
CONFIGS = {
    'rf_baseline': {'n_estimators': 100, 'max_depth': 5},      # базова конфігурація з методичних рекомендацій
    'rf_full': {'n_estimators': 300, 'max_depth': None},       # повні дерева (найкращий алгоритм для DTD у ПР-5)
}
RESULTS = Path('results')


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def git_commit():
    """Коміт коду: з аргументу збірки Docker (GIT_COMMIT) або з локального репозиторію."""
    if os.environ.get('GIT_COMMIT', 'unknown') != 'unknown':
        return os.environ['GIT_COMMIT']
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def runtime():
    return 'docker' if Path('/.dockerenv').exists() else 'host'


def run(name, X_train, X_val, y_train, y_val, data_md5, commit):
    params = {**CONFIGS[name], 'random_state': SEED, 'test_size': 0.2, 'model': 'RandomForestClassifier'}
    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params(params)
        mlflow.set_tags({'dataset': 'DTD-weave: woven, knitted, braided, meshed (480 images, 104 features)',
                         'dataset_md5': data_md5, 'git_commit': commit, 'runtime': runtime(),
                         'python': platform.python_version(), 'platform': platform.platform(),
                         'sklearn': sklearn.__version__, 'numpy': np.__version__})

        model = RandomForestClassifier(**CONFIGS[name], random_state=SEED, n_jobs=1)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        metrics = {'f1_score': f1_score(y_val, y_pred, average='weighted'),
                   'f1_macro': f1_score(y_val, y_pred, average='macro'),
                   'accuracy': accuracy_score(y_val, y_pred),
                   'roc_auc_ovr': roc_auc_score(y_val, proba, multi_class='ovr', labels=model.classes_)}
        mlflow.log_metrics({**metrics, 'train_time_s': train_time})

        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, labels=model.classes_, cmap='Blues', ax=ax,
                                                colorbar=False)
        ax.set_title(f'{name}: confusion matrix (validation)')
        fig.tight_layout()
        mlflow.log_figure(fig, 'confusion_matrix.png')
        plt.close(fig)
        mlflow.log_dict(classification_report(y_val, y_pred, output_dict=True), 'classification_report.json')
        mlflow.sklearn.log_model(model, 'random_forest_model', signature=infer_signature(X_val, y_pred),
                                 input_example=X_val.iloc[:3])

        print(f'{name:12s} run_id={active.info.run_id}  f1_weighted={metrics["f1_score"]:.6f}  '
              f'f1_macro={metrics["f1_macro"]:.6f}  accuracy={metrics["accuracy"]:.6f}  '
              f'roc_auc={metrics["roc_auc_ovr"]:.6f}  train_time={train_time:.2f}s')
        return {k: round(float(v), 10) for k, v in metrics.items()}


def compare_with_reference(results):
    """Перевірка відтворюваності: метрики якості мають збігатися з еталонними до 1e-9."""
    ref_path = RESULTS / 'reference_metrics.json'
    if not ref_path.exists():
        print('еталонних метрик немає – порівняння пропущено')
        return True
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    ok = True
    for name, metrics in results.items():
        for key, value in metrics.items():
            expected = ref.get('runs', {}).get(name, {}).get(key)
            if expected is None or abs(expected - value) > 1e-9:
                ok = False
                print(f'  РОЗБІЖНІСТЬ {name}.{key}: {value} (еталон {expected})')
    print('ВІДТВОРЕНО: метрики збігаються з еталонними' if ok else 'НЕ ВІДТВОРЕНО: є розбіжності з еталоном')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--runs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument('--strict', action='store_true', help='код виходу 1, якщо метрики не збігаються з еталоном')
    args = parser.parse_args()

    if not Path(args.data).exists():               # дані не отримано через DVC – відтворюємо з першоджерела
        print(f'{args.data} не знайдено – запуск prepare.py')
        import prepare
        prepare.main()

    df = pd.read_csv(args.data)
    X, y = df.drop(columns=['image', 'label']), df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    data_md5, commit = md5(args.data), git_commit()
    print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
    print(f'dataset {args.data}: {len(df)} x {X.shape[1]}, MD5 {data_md5}; commit {commit[:12]}; runtime {runtime()}')

    mlflow.set_experiment(EXPERIMENT)
    results = {name: run(name, X_train, X_val, y_train, y_val, data_md5, commit) for name in args.runs}

    RESULTS.mkdir(exist_ok=True)
    summary = {'runtime': runtime(), 'python': platform.python_version(), 'platform': platform.platform(),
               'dataset_md5': data_md5, 'git_commit': commit, 'runs': results}
    (RESULTS / f'metrics_{runtime()}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                       encoding='utf-8')
    ok = compare_with_reference(results)
    if args.strict and not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
