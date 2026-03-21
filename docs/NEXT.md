# Дальнейшие шаги

## Улучшение MSE

- Реализовано: `te_freq`, редкие категории → `__OTHER__` (`pipeline.rare_category_min_count`), CV / full-train, **сетка + Cartesian**, **Optuna** (`hparam_search.mode: optuna`), **K-fold blend** на test (`evaluation.ensemble`), CatBoost.
- Дальше: стекинг по OOF, ансамбли разных семейств, псевдо-лейблы.

Новые эксперименты — в [docs/RESULTS.md](RESULTS.md) и `artifacts/experiments.jsonl`.

## Развитие системы

- **RAG:** `knowledge/`, `scripts/build_rag_index.py`, Explorer; сравнение — `scripts/compare_architectures.py`.
- **Память экспериментов:** `artifacts/experiments.jsonl` — в т.ч. `agent_metrics`, `hparam_mode` (`grid` / `optuna`), `hparam_*`.
- **Builder:** Critic, детерминированный финал, опционально **ensemble.kfold_blend_test**.
- **Benchmarking ML:** `scripts/run_benchmark.py`.

Подробнее — [ARCHITECTURE.md](../ARCHITECTURE.md).
