# Мультиагентная система для Kaggle-регрессии (MSE)

Проект: автоматизация полного цикла EDA → feature engineering → обучение модели → submission для соревнования с метрикой **MSE**.

- **Соревнование:** [пригласительная ссылка](https://www.kaggle.com/t/88dbc788cead475494fc76413a69f7ee)
- **Метрика:** MSE (Mean Squared Error)
- **Baseline с лидерборда:** 10986.9346
- **Лучший результат:** Val MSE 10578.78 (LightGBM, `n_estimators=300`, `max_depth=8`)
- **Данные:** табличный датасет (train.csv, test.csv), целевая переменная `target`

---

## Документация

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — архитектура агентов, RAG, инструменты, безопасность, оценка
- **[docs/RESULTS.md](docs/RESULTS.md)** — таблица экспериментов (модель, параметры, Val MSE, команды)
- **[docs/NEXT.md](docs/NEXT.md)** — дальнейшие шаги по улучшению и развитию системы

---

## Структура проекта

- `config/` — настройки ([settings.yaml](config/settings.yaml)), системные промпты агентов
- `src/agents/` — Explorer, Engineer, Builder, Coordinator
- `src/tools/` — инструменты EDA, препроцессинга, обучения (Tool Use)
- `src/rag/` — индексация и поиск по базе знаний (Chroma + sentence-transformers)
- `src/memory/` — журнал экспериментов (`artifacts/experiments.jsonl`, в т.ч. токены LLM и метки прогонов)
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install xgboost lightgbm   # опционально, для бустинга
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

Если документов в `knowledge/` нет или RAG выключен (`rag.enabled: false`), пайплайн идёт без RAG-контекста.

### 4. Мультиагентный пайплайн (нужен API-ключ)

```bash
source .venv/bin/activate
python run.py
# или: python run.py --data-dir data --config config/settings.yaml
```

Цепочка: **Explorer → Engineer → Builder**. В stderr пишутся логи вызовов инструментов. После успешного прогона в `artifacts/experiments.jsonl` добавляется строка с `val_mse` и путём к submission.

Параметры автоматического прогона — в [config/settings.yaml](config/settings.yaml):

- **`pipeline`**: `encoding` (`te_freq` или `ohe`), опционально **`rare_category_min_count`** (редкие категории → `__OTHER__` при `te_freq`).
- **`evaluation`**: `val_ratio`, `cv_folds`, `fit_full_train`, **`hparam_search`** (`mode: grid` или **`optuna`**), **`ensemble`** (K-fold blend предсказаний на test); при **`use_best_only: true`** финальный шаг **без LLM** Builder.
- **`robustness`**: по умолчанию клип предсказаний **выключен** (`clip_predictions: false`); при необходимости включите для борьбы с выбросами.
- **`agents.builder`**: `critic_max_iterations`, **`mse_threshold`** (ранний выход из цикла Critic при LLM Builder), **`critic_iterate_without_threshold`** (несколько раундов LLM даже без порога).

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

### 6. Сравнение нескольких конфигураций (бенчмарк без LLM)

Скрипт гоняет несколько моделей подряд, складывает submission в `submissions/submission_benchmark_0.csv`, …

```bash
python scripts/run_benchmark.py
```

Набор конфигураций редактируется в [scripts/run_benchmark.py](scripts/run_benchmark.py) (список `configs`).

### 7. Сравнение сценариев (RAG / без RAG и др.)

Подряд запускаются два конфига (по умолчанию основной и `config/settings.norag.yaml`), копии submission складываются в `submissions/submission_arch_*.csv`:

```bash
python scripts/compare_architectures.py
```

Подробности — [docs/RESULTS.md](docs/RESULTS.md) (раздел про архитектуры).

---

## Программный API

- Журнал экспериментов: `src.memory.experiments.save_experiment`, `list_experiments` (поля `agent_metrics`, `hparam_*`, `experiment_label` при наличии)
- Перебор гиперпараметров: `src.evaluation.hparam_search` (`build_search_grid`, `run_hparam_grid`, `select_best_with_policy`)
- Бенчмарк ML: `src.evaluation.benchmark.run_benchmark(configs, data_dir, artifacts_dir, submissions_dir)`
- RAG: `src.rag.indexer.build_index`, `src.rag.retriever.retrieve`

---

## Прочее

Артефакты и сгенерированные submission не коммитятся (см. `.gitignore`). Файл `submissions/submission.csv` загружается на страницу соревнования Kaggle.
