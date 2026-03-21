# Дальнейшие шаги

## Улучшение MSE

- Реализовано: `te_freq`, редкие категории → `__OTHER__` (`pipeline.rare_category_min_count`), CV / full-train, **сетка + Cartesian**, **Optuna** (`hparam_search.mode: optuna`), **K-fold blend** на test (`evaluation.ensemble`), CatBoost, **OOF-стекинг** (`evaluation.stacking` + `src/tools/advanced_ensemble.py`), **псевдо-лейблы** (`evaluation.pseudo_labels`).
- Дальше: подбор `base_models` / меты под датасет, отчёт по стекингу в `experiments.jsonl`.

Новые эксперименты — в [docs/RESULTS.md](RESULTS.md) и `artifacts/experiments.jsonl`.

## Развитие системы

- **Соответствие курсу / отчёт:** [COMPLIANCE.md](COMPLIANCE.md).
- **RAG:** `knowledge/`, `scripts/build_rag_index.py`, Explorer; сравнение — `scripts/compare_architectures.py`.
- **Память экспериментов:** `artifacts/experiments.jsonl` — в т.ч. `agent_metrics`, `artifact_validation`, `hparam_mode`.
- **Feedback:** валидация CSV после Engineer (`artifact_validation`), Critic у Builder, Optuna/сетка.
- **Benchmarking ML:** `scripts/run_benchmark.py`.

Подробнее — [ARCHITECTURE.md](../ARCHITECTURE.md).
