# Результаты экспериментов (MSE)

Метрика соревнования: **MSE** (Mean Squared Error).  
Baseline с лидерборда: **10986.9346** — целевое значение для улучшения.

Условия: один и тот же препроцессинг (`src/tools/feature_tools.py`), разбиение train 80% / 20% для валидации, `random_state=42`.

---

## Сводная таблица

| № | Модель        | Параметры                          | Val MSE     | Baseline побит | Команда |
|---|---------------|------------------------------------|-------------|----------------|---------|
| 1 | Ridge         | по умолчанию                      | 13168.4755  | нет            | `python scripts/run_pipeline_manual.py --data-dir data --model ridge` |
| 2 | Random Forest | по умолчанию                      | 10961.0854  | да             | `python scripts/run_pipeline_manual.py --data-dir data --model random_forest` |
| 3 | XGBoost       | по умолчанию                      | 10834.3184  | да             | `python scripts/run_pipeline_manual.py --data-dir data --model xgboost` |
| 4 | LightGBM      | по умолчанию                      | 10815.4563  | да             | `python scripts/run_pipeline_manual.py --data-dir data --model lightgbm` |
| 5 | LightGBM      | `n_estimators=300`, `max_depth=8`  | **10578.7835** | да          | `python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --n-estimators 300 --max-depth 8` |
| 6 | XGBoost       | `n_estimators=400`, `learning_rate=0.05` | 10726.4128 | да        | `python scripts/run_pipeline_manual.py --data-dir data --model xgboost --learning-rate 0.05 --n-estimators 400` |

**Лучший результат:** № 5 — LightGBM, Val MSE **10578.7835**.

---

## Воспроизведение

1. Данные в `data/train.csv`, `data/test.csv`.
2. Зависимости: `pip install -r requirements.txt` (для XGBoost/LightGBM: `pip install xgboost lightgbm`).
3. Запуск команды из таблицы. Val MSE выводится в консоль; submission — в `submissions/submission.csv`.

Новые эксперименты добавляются в таблицу: модель, параметры, Val MSE, команда.
