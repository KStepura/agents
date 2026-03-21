# Результаты экспериментов (MSE)

Метрика соревнования: **MSE** (Mean Squared Error).  
Baseline с лидерборда: **10986.9346** — целевое значение для улучшения.

Условия по умолчанию: препроцессинг `src/tools/feature_tools.py`, разбиение train 80% / 20% для валидации, `random_state=42`.

---

## Режимы препроцессинга

| Режим | Описание |
|-------|----------|
| `te_freq` (по умолчанию) | Target encoding + log-frequency по категориям, признаки месяц/день недели для `last_dt`, без OHE — компактная матрица, быстрее по времени. |
| `ohe` | Legacy: OneHotEncoder + StandardScaler (очень много колонок). |

Флаг CLI: `--encoding te_freq` или `--encoding ohe` в `scripts/run_pipeline_manual.py`.

---

## Оценка качества

| Опция | Описание |
|-------|----------|
| Holdout (по умолчанию) | Val MSE на 20% holdout. |
| `--cv-folds K` | Дополнительно: K-fold CV на train (среднее и std MSE). |
| `--full-train` | Обучение на **100%** train для финального `submission.csv` (рекомендуется для Kaggle после выбора гиперпараметров). Сочетайте с `--cv-folds` для отчёта о качестве без отдельного holdout. |

---

## Сводная таблица (исторические + обновлённые прогоны)

| № | Препроцессинг | Модель | Параметры | Val MSE | Примечание |
|---|---------------|--------|-----------|---------|------------|
| A | `te_freq` | RandomForest | default | **7183.52** | `python scripts/run_pipeline_manual.py --encoding te_freq --model random_forest` |
| B | `ohe` | LightGBM | n=300, depth=8 | 10578.78 | прежний лучший с OHE (см. архив ниже) |
| 1 | `ohe` | Ridge | default | 13168.48 | |
| 2 | `ohe` | Random Forest | default | 10961.09 | |
| 3 | `ohe` | XGBoost | default | 10834.32 | |
| 4 | `ohe` | LightGBM | default | 10815.46 | |
| 5 | `ohe` | LightGBM | n=300, depth=8 | 10578.78 | прежний лучший с OHE |
| 6 | `ohe` | XGBoost | lr=0.05, n=400 | 10726.41 | |

**Примечание:** Val MSE на одном holdout **не сравним напрямую** между разными моделями без фиксированного протокола; для сравнения используйте `--cv-folds 5` на одном и том же `encoding`.

**Примеры команд:**

```bash
# Быстрый прогон с te_freq (по умолчанию)
python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --n-estimators 300 --max-depth 8

# K-fold + обучение на всём train
python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --cv-folds 5 --full-train --n-estimators 300 --max-depth 8

# CatBoost (нужен: pip install catboost)
python scripts/run_pipeline_manual.py --model catboost --n-estimators 500 --max-depth 8
```

Журнал прогонов также пишется в `artifacts/experiments.jsonl` (из `run.py` и `run_pipeline_manual.py`).

---

## Автоматический прогон (`python run.py`)

Единственная точка входа полного цикла (Explorer → фичи → Optuna/сетка при необходимости → Builder). Скрипты в `scripts/` (`run_benchmark.py`, `run_pipeline_manual.py`, `compare_architectures.py`) запускаются **отдельно** и не подключаются к `run.py`.

| Опция в `config/settings.yaml` | Описание |
|--------------------------------|----------|
| `evaluation.hparam_search` | Перебор конфигураций после подготовки данных (LLM Engineer или `preprocessing_search`), выбор лучшего по Val/CV MSE → Builder. |
| `hparam_search.mode` | `grid` (сетка + cartesian) или `optuna` (Bayesian optimization; `pip install optuna`, см. `optuna:` в конфиге). |
| `hparam_search.grid` | Явный список словарей `{ model, n_estimators, max_depth, learning_rate, … }`. |
| `hparam_search.cartesian` | Одна `model` + оси-списки; декартово произведение **дополняет** `grid` (сначала строки `grid`, затем развёртка). |
| `hparam_search.eval_cv_folds` | Число фолдов **только для поиска** (3 — быстрее, 5 — стабильнее оценка MSE); финальная модель использует `evaluation.cv_folds`. |
| `hparam_search.preprocessing_search` | `enabled: true`: без LLM Engineer — перебор `variants` (encoding, `rare_category_min_count`), на каждом **Optuna**; лучший вариант копируется в `train_processed.csv`. Требует `mode: optuna`. |
| `hparam_search.optuna.models` | Список из 2+ моделей (например `[lightgbm, catboost]`): каждый trial выбирает семейство + гиперпараметры. |
| `hparam_search.optuna.lgbm_regularization` | `true`: в поиске LightGBM дополнительно `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`. |
| `hparam_search.max_trials` | Максимум комбинаций после объединения `grid` + `cartesian` (обрезка с начала списка). |
| `hparam_search.selection_policy` | `prefer_boosting: true` — если по CV выиграл Ridge, но есть LightGBM/RF/XGB/CatBoost с MSE не хуже чем (1 + `tie_relative_tolerance`) × лучший MSE, выбирается бустинг. |
| `hparam_search.use_best_only: true` | Финальное обучение и `submission.csv` **без LLM** Builder. `false` — LLM Builder + подсказка + Critic. |
| `agents.builder.mse_threshold` | Порог для раннего выхода из Critic (только LLM Builder). |
| `robustness.*` | Клип предсказаний / фич; по умолчанию `clip_predictions: false` (часто лучше для LB). |
| `evaluation.stacking` | `enabled` + непустой `base_models`: OOF-предсказания баз + мета-модель (`meta`) на OOF; при `use_best_only` приоритет над blend и pseudo. См. `src/tools/advanced_ensemble.py`. |
| `evaluation.pseudo_labels` | `enabled`: итерации дообучения с добавлением test с предсказанным target; финальный сабмит — модель после последнего раунда. Модель/параметры из `hparam_best`. |
| `evaluation.ensemble` | `enabled` + `kfold_blend_test`: K моделей на train (каждая на K−1 фолде), **среднее предсказаний на test** (вместо одной модели на 100% train). |
| `agents.engineer.artifact_validation` | **Feedback loop:** CSV (`artifact_validation.py`); опционально `check_python_scripts`, `check_preprocessor_joblib` (`code_validation.py`); при ошибке — повтор Engineer (`max_rounds`, `strict`). |
| `monitoring.run_summary_json` | `true` → **`artifacts/run_summary.json`**: длительность прогона, val_mse, `agent_metrics`, `artifact_validation`. |
| `pipeline.rare_category_min_count` | Для `te_freq`: значения категории с частотой &lt; порога на train заменяются на **`__OTHER__`**; на test редкие/новые уровни тоже → `__OTHER__`. |

LightGBM в сетке поддерживает **`num_leaves`**, **`min_child_samples`** (см. `src/security/validation.py`).

Соответствие требованиям курса и обоснование решений — **[docs/COMPLIANCE.md](COMPLIANCE.md)**.

В `experiments.jsonl` для `run.py` дополнительно: `agent_metrics`, `artifact_validation`, `hparam_*`, `hparam_mode`, `builder_mode`, опционально `experiment_label`.

---

## Сравнение сценариев (мультиагентная «архитектура», не только ML)

Скрипт [scripts/compare_architectures.py](../scripts/compare_architectures.py) запускает два конфига подряд (по умолчанию `config/settings.yaml` с RAG и `config/settings.norag.yaml` без RAG), сохраняет копии submission в `submissions/submission_arch_rag_on.csv` и `submission_arch_rag_off.csv`, метки — в `experiments.jsonl` (`experiment_label`).

| Сценарий | Что отличается |
|----------|------------------|
| RAG вкл | Explorer получает фрагменты из `knowledge/` при непустом индексе Chroma. |
| RAG выкл | Тот же пайплайн, но без RAG-контекста в Explorer. |

Остальное (препроцессинг, сетка гиперпараметров, robustness) держите **одинаковым** в обоих YAML для честного сравнения.

---

## Воспроизведение

1. Данные в `data/train.csv`, `data/test.csv`.
2. Зависимости: `pip install -r requirements.txt` (для бустинга: `pip install xgboost lightgbm catboost`).
3. Запуск команд из таблицы. Val MSE выводится в консоль; submission — в `submissions/submission.csv`.

Новые эксперименты добавляйте в таблицу: препроцессинг, модель, параметры, Val MSE / CV MSE, команда.
