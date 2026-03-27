# Мультиагентная система для Kaggle-регрессии (MSE)

Проект: автоматизация полного цикла EDA → feature engineering → обучение модели → submission для соревнования с метрикой **MSE**.

- **Соревнование:** [пригласительная ссылка](https://www.kaggle.com/t/88dbc788cead475494fc76413a69f7ee)
- **Метрика:** MSE (Mean Squared Error)
- **Baseline (лидерборд):** 10986.9346
- **Лучший прогон (по `config/settings.yaml`):** 10205.0493
- **Лучший прогон без RAG (по `config/settings.norag.yaml`):** 10578.7835
- **По умолчанию** (`config/settings.yaml`): два варианта **te_freq** (разный `rare_category_min_count`) + Optuna с **LightGBM и CatBoost**; режим **ohe** в список вариантов не входит (слишком широкая матрица, прогон может «зависнуть»). LLM Engineer не вызывается — см. `preprocessing_search`. Для цепочки с Engineer выключите `evaluation.hparam_search.preprocessing_search.enabled`.
- **Данные:** табличный датасет (train.csv, test.csv), целевая переменная `target`

---

## Документация

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — архитектура агентов, RAG, инструменты, безопасность, **feedback loops**, оценка
- **[docs/COMPLIANCE.md](docs/COMPLIANCE.md)** — **соответствие требованиям курса**, обоснование решений, ограничения (для отчёта и проверки)
- **[docs/RESULTS.md](docs/RESULTS.md)** — таблица экспериментов (модель, параметры, Val MSE, команды)

---

## Структура проекта

- `config/` — настройки ([settings.yaml](config/settings.yaml)), системные промпты агентов
- `src/agents/` — Explorer, Engineer, Builder, Coordinator
- `src/tools/` — инструменты EDA, препроцессинга, обучения (Tool Use)
- `src/rag/` — индексация и поиск по базе знаний (Chroma + sentence-transformers)
- `src/memory/` — журнал экспериментов (`artifacts/experiments.jsonl`, в т.ч. токены LLM, `artifact_validation`, метки прогонов)
- `src/evaluation/` — бенчмарк, Optuna/сетка гиперпараметров, **вложенный поиск по препроцессингу** (`preprocessing_search`), **проверка артефактов Engineer**
- `src/security/` — проверка путей, whitelist инструментов, границы гиперпараметров
- `src/monitoring/` — логирование вызовов инструментов (stderr)
- `knowledge/` — тексты для RAG (Markdown / `.txt`)
- `data/` — train.csv, test.csv
- `artifacts/` — препроцессор, индекс RAG, обработанные данные, модели (генерируются, в `.gitignore`)
- `submissions/` — submission.csv для загрузки на Kaggle

---

## Быстрый старт

### 1. Окружение (рекомендуется venv)

На macOS с системным Python часто действует PEP 668 — проще создать виртуальное окружение:

```bash
cd /path/to/agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env` и задайте `OPENROUTER_API_KEY` для запуска с LLM.

### 2. Данные

В каталоге `data/` должны быть `train.csv` и `test.csv` (как в задании соревнования).

### 3. RAG (база знаний для Explorer)

При **`python run.py`** индекс Chroma **собирается автоматически**, если в [config/settings.yaml](config/settings.yaml) включено `rag.enabled`, каталог `knowledge/` содержит `.md`/`.txt`, а каталог индекса (`rag.persist_path`, по умолчанию `artifacts/chroma_rag`) ещё пуст или отсутствует. Первый такой запуск может долго идти из‑за скачивания модели эмбеддингов (нужен интернет).

Пересобрать индекс вручную после правок в `knowledge/`:

```bash
python scripts/build_rag_index.py
```

По умолчанию в основном автопрогоне RAG включен (`config/settings.yaml`), и Explorer получает контекст из `knowledge/`, если индекс доступен.

### 4. Мультиагентный пайплайн (нужен API-ключ)

```bash
source .venv/bin/activate
python run.py
```

**Что делает `run.py` (автоматический прогон):** Coordinator вызывает Explorer (EDA + RAG по основному конфигу), затем либо **вложенный препроцессинг + Optuna** (`evaluation.hparam_search.preprocessing_search.enabled: true` — детерминированные фичи, **без LLM Engineer**), либо **Engineer** (LLM + инструменты `fit_preprocessor` / `transform`), затем **подбор гиперпараметров** (Optuna или сетка, если включено), затем **Builder** (при `use_best_only: true` — детерминированное обучение и `submission.csv` без LLM). В stderr — логи вызовов инструментов; в `artifacts/experiments.jsonl` и при `monitoring.run_summary_json` — в `artifacts/run_summary.json` — метрики и длительность.

Параметры — в [config/settings.yaml](config/settings.yaml):

- **`pipeline`**: базовые значения для Engineer; при `preprocessing_search` перебираются варианты из `variants`.
- **`evaluation`**: `val_ratio`, `cv_folds`, `fit_full_train`, **`hparam_search`** (Optuna / сетка, **`preprocessing_search`**, **`ensemble`**, опционально **`stacking`** / **`pseudo_labels`** — код в `src/tools/advanced_ensemble.py`, подгружается только если включено); при **`use_best_only: true`** финальный Builder **без LLM**.
- **`monitoring`**: `run_summary_json` — сводка прогона в `artifacts/run_summary.json`.
- **`robustness`**: клип предсказаний по умолчанию выключен (`clip_predictions: false`).
- **`agents.builder`**: настройки Critic, если LLM Builder используется (`use_best_only: false`).

Флаг **`python run.py --experiment-label NAME`** добавляет метку в строку `experiments.jsonl` (удобно для [scripts/compare_architectures.py](scripts/compare_architectures.py)).

### 5. Без LLM (воспроизводимый бенчмарк одной модели)

По умолчанию препроцессинг **`te_freq`** (target + frequency encoding, без огромного OHE). Старый режим: `--encoding ohe`.

```bash
python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --n-estimators 300 --max-depth 8
```

K-fold оценка и обучение на **всём** train для сабмита:

```bash
python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --cv-folds 5 --full-train --n-estimators 300 --max-depth 8
```

### 6. Сравнение нескольких конфигураций (вспомогательный скрипт, не `run.py`)

`scripts/run_benchmark.py` гоняет несколько моделей подряд на одном препроцессинге, складывает submission в `submissions/submission_benchmark_0.csv`, … Редактируйте список `configs` внутри скрипта.

```bash
python scripts/run_benchmark.py
```

### 7. Сравнение сценариев (RAG / без RAG и др.)

Подряд запускаются два конфига (по умолчанию основной и `config/settings.norag.yaml`), копии submission складываются в `submissions/submission_arch_*.csv`:

```bash
python scripts/compare_architectures.py
```

Подробности — [docs/RESULTS.md](docs/RESULTS.md) (раздел про архитектуры).

---

## Программный API (для расширений и тестов)

- Журнал экспериментов: `src.memory.experiments.save_experiment`, `list_experiments`
- Перебор гиперпараметров: `src.evaluation.hparam_search`, `src.evaluation.optuna_hparam`, `src.evaluation.preprocessing_search`
- Опциональные режимы Builder: `src.tools.advanced_ensemble` (стекинг / псевдо-лейблы), только если включено в `evaluation`
- RAG: `src.rag.indexer.build_index`, `src.rag.retriever.retrieve`
- Вспомогательный бенчмарк нескольких моделей: функция `run_benchmark` в [scripts/run_benchmark.py](scripts/run_benchmark.py)

---

## Прочее

Артефакты и сгенерированные submission не коммитятся (см. `.gitignore`). Файл `submissions/submission.csv` загружается на страницу соревнования Kaggle.
