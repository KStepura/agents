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

- `config/` — настройки (settings.yaml), системные промпты агентов
- `src/agents/` — Explorer, Engineer, Builder, Coordinator
- `src/tools/` — инструменты EDA, препроцессинга, обучения (Tool Use)
- `src/rag/`, `src/memory/`, `src/security/`, `src/evaluation/` — RAG, память, безопасность, метрики
- `data/` — train.csv, test.csv
- `artifacts/` — препроцессор, обработанные данные, модели (генерируются, в git не коммитятся)
- `submissions/` — submission.csv для загрузки на Kaggle

---

## Запуск

### Требования

- Python 3.10+
- Данные в `data/train.csv`, `data/test.csv`
- Зависимости: `pip install -r requirements.txt` (для XGBoost/LightGBM: `pip install xgboost lightgbm`)

### С агентами (LLM)

В `.env` задаётся `OPENROUTER_API_KEY` (https://openrouter.ai/). Затем:

```bash
python run.py
```

Пайплайн: Explorer → Engineer → Builder (EDA, препроцессинг, обучение, submission).

### Без агентов (ручной выбор модели)

Лучший зафиксированный результат (Val MSE 10578.78):

```bash
python scripts/run_pipeline_manual.py --data-dir data --model lightgbm --n-estimators 300 --max-depth 8
```

Другие модели и параметры: `--model ridge`, `random_forest`, `xgboost`, `lightgbm`; опции `--n-estimators`, `--max-depth`, `--learning-rate`, `--alpha`. Полный список экспериментов — в [docs/RESULTS.md](docs/RESULTS.md).

Артефакты и submission создаются скриптом; в репозиторий не коммитятся (см. `.gitignore`). Файл `submissions/submission.csv` загружается на страницу соревнования Kaggle.

