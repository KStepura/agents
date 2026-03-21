# Архитектура мультиагентной системы для Kaggle-регрессии (MSE)

## Контекст

- **Соревнование**: регрессия, метрика **MSE** (Mean Squared Error).
- **Данные**: табличный датасет (train.csv, test.csv), колонки: name, _id, host_name, location_cluster, location, lat, lon, type_house, sum, min_days, amt_reviews, last_dt, avg_reviews, total_host, **target**.
- **Цель**: автоматизировать полный цикл от EDA до обучения модели и формирования submission.

---

## Общая схема

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    COORDINATOR                           │
                    │  (планирование, маршрутизация, финальные решения)       │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌─────────────────┐               ┌─────────────────┐               ┌─────────────────┐
│    EXPLORER     │               │   ENGINEER      │               │    BUILDER      │
│  (EDA, отчёты)  │──────────────▶│ (фичи, препроцесс)│──────────────▶│ (модель, MSE)   │
│  + RAG (знания) │               │  + Tool Use      │               │  + Critic / hparam │
└─────────────────┘               └─────────────────┘               └─────────────────┘
         │                                    │                                    │
         └────────────────────────────────────┼────────────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  EVALUATION + GUARDRAILS + MEMORY (эксперименты)        │
                    └─────────────────────────────────────────────────────────┘
```

---

## Feedback loops (обратная связь в цепочке)

Требование курса: в агентной архитектуре должен быть **цикл улучшения** по результату предыдущего шага. В проекте это не «исполнение произвольного кода», а **итерации с проверкой артефактов и метрик**:

1. **Engineer → проверка артефактов** ([`src/evaluation/artifact_validation.py`](src/evaluation/artifact_validation.py), [`src/evaluation/code_validation.py`](src/evaluation/code_validation.py)): после вызова Engineer Coordinator проверяет CSV (файлы, `target`, согласованность колонок), при необходимости — **синтаксис `*.py`** в `artifacts/` и **корректность `preprocessor.joblib`** (объект `sklearn.pipeline.Pipeline` в сохранённом словаре). При ошибке Engineer вызывается **повторно** с явным текстом замечаний (`agents.engineer.artifact_validation`). Токены LLM по раундам **суммируются** в `llm_usage`.
2. **Builder (LLM-режим)** ([`src/agents/builder.py`](src/agents/builder.py)): **Critic** — если MSE выше порога или задано несколько раундов, следующий user-message содержит подсказку сменить модель/гиперпараметры.
3. **Подбор модели**: Optuna / сетка + политика `prefer_boosting` — обратная связь по **метрике CV**, а не по тексту.

Подробная привязка к формулировкам задания и ограничения — в **[docs/COMPLIANCE.md](docs/COMPLIANCE.md)**.

---

## Агенты (минимум 3, по требованиям курса)

### 1. Explorer (Исследователь данных)

- **Роль**: EDA, выявление выбросов, пропусков, распределений, корреляций с `target`.
- **Паттерн**: ReAct (Reasoning + Act) — рассуждение и вызов инструментов по очереди.
- **Инструменты (Tool Use)**:
  - загрузка и обзор `train`/`test`;
  - описательная статистика, типы колонок;
  - визуализации (опционально — сохранение графиков);
  - отчёт в структурированном виде (JSON/текст) для Engineer.
- **RAG**: база знаний по типовым приёмам EDA для регрессии и по успешным Kaggle-решениям (если добавлены документы).
- **Выход**: артефакт «EDA report» (путь к отчёту или структура в памяти).

### 2. Engineer (Инженер признаков)

- **Роль**: препроцессинг, кодирование категорий, работа с пропусками, генерация/отбор признаков на основе отчёта Explorer.
- **Паттерн**: Chain-of-Thought + Tool Use — пошаговые рассуждения и вызов скриптов.
- **Инструменты**:
  - загрузка данных и EDA-отчёта;
  - создание pipeline (например, sklearn Pipeline) или скриптов препроцессинга;
  - feature selection (фильтрация, отбор по важности);
  - сохранение обработанных датасетов (train/val/test) для Builder.
- **Вход**: артефакт от Explorer.
- **Выход**: артефакт «feature pipeline» + пути к обработанным данным.

### 3. Builder (Построитель модели)

- **Роль**: выбор регрессора, кросс-валидация, минимизация MSE, формирование предсказаний и submission.
- **Паттерн**: Planner–Executor–Critic: план → выполнение (`train_regressor`, `make_submission`) → при невысоком MSE или по порогу — остановка; иначе повтор с подсказкой Critic (несколько раундов LLM, если задан `mse_threshold` или `critic_iterate_without_threshold`). Если в конфиге включён **`evaluation.hparam_search`** с **`use_best_only: true`**, финальное обучение и submission выполняются **детерминированно** по лучшей точке сетки (LLM Builder не вызывается); при этом приоритет: **`evaluation.stacking`** (OOF + мета) → **`evaluation.pseudo_labels`** → **K-fold blend** (`evaluation.ensemble`) → одиночная модель — см. [`src/tools/advanced_ensemble.py`](src/tools/advanced_ensemble.py).
- **Инструменты**:
  - загрузка обработанных данных;
  - обучение регрессоров (например, Ridge, RandomForest, XGBoost/LightGBM, простые нейросети);
  - расчёт MSE на валидации;
  - генерация `submission.csv` в формате соревнования (index, prediction).
- **Вход**: артефакты от Engineer.
- **Выход**: путь к submission, метрики (MSE), краткое обоснование выбора модели.

### 4. Coordinator

- **Роль**: оркестрация шагов Explorer → Engineer → (опционально **перебор гиперпараметров**: сетка `grid` + `cartesian`, или **Optuna**, политика `selection_policy`) → Builder и передача артефактов. Опционально **`preprocessing_search`** ([`preprocessing_search.py`](src/evaluation/preprocessing_search.py)): вместо LLM Engineer — перебор вариантов препроцессинга с Optuna на каждом, затем Builder.
- **Паттерн**: Supervisor.
- **Реализация в коде**: [`coordinator.py`](src/agents/coordinator.py) — без отдельного LLM; после Engineer (или вложенного препроцессинга) — **валидация артефактов** и при необходимости повтор Engineer; затем [`run_hparam_grid`](src/evaluation/hparam_search.py) или **Optuna**, если включён поиск и не использован только что вложенный Optuna; агрегирует **`agent_metrics`** для `experiments.jsonl`.

---

## RAG-модули

- **Назначение**: дать агентам доступ к лучшим практикам (регрессия, MSE, табличные данные, Kaggle).
- **Компоненты**:
  - **Векторная БД**: ChromaDB или FAISS.
  - **Embedding-модель**: через OpenRouter или локально (e.g. sentence-transformers).
  - **Корпус**: документы/страницы по EDA, feature engineering, регрессии, формату submission; при наличии — фрагменты кода или описания решений с Kaggle.
- **Интеграция**: Explorer и при необходимости Engineer запрашивают RAG при формировании плана и отчёта.

---

## Tool Use и интерфейсы

- Все инструменты — **безопасные** Python-функции с чёткими входами/выходами.
- **Input validation**: проверка путей к файлам (только разрешённые директории), типов и диапазонов параметров (например, `n_estimators`, `max_depth`).
- **MCP (Model Context Protocol)**: при желании обернуть набор инструментов (EDA, feature selection, обучение) в MCP-сервер для единообразного доступа агентов.
- Примеры инструментов (см. `src/agents/tools_for_llm.py`):
  - Explorer: `load_and_summarize`, `get_missing_and_correlation`, `save_eda_report`
  - Engineer: `fit_preprocessor`, `transform_train_test`
  - Builder: `train_regressor` (MSE на валидации в ответе), `make_submission`

---

## Память и персистентность

- **Краткосрочная**: контекст диалога внутри одного запуска (история сообщений агентов).
- **Долговременная**:
  - **Эксперименты**: логи запусков (какие фичи, какая модель, MSE), сохраняемые в JSON/SQLite или простой БД.
  - **Артефакты**: EDA-отчёты, датасеты после препроцессинга, обученные модели, submission-файлы — в каталогах `artifacts/`, `submissions/`.
- Это даёт воспроизводимость и возможность сравнения конфигураций (benchmarking).

---

## Безопасность и надёжность

- **Input validation**: все пути — нормализованные и проверенные (no path traversal); параметры моделей — в белом списке или в ограниченных диапазонах.
- **Guardrails**: 
  - запрет выполнения произвольного кода от LLM (только вызов заранее объявленных инструментов);
  - при использовании `exec`/`eval` — только в sandbox или запрет.
- **Мониторинг**: логирование вызовов инструментов (время), метрик MSE; в `run.py` — сводка **токенов и числа вызовов API** по агентам в `experiments.jsonl` (`agent_metrics`); опционально **`artifacts/run_summary.json`** (`monitoring.run_summary_json`) — длительность прогона и ключевые поля результата.
- **Устойчивость выдачи** (минимальный уровень для отчёта): в [`model_tools.make_submission`](src/tools/model_tools.py) опционально **клип предсказаний** по квантилям `target` на train и опционально **ограничение числовых признаков теста** диапазоном train (см. `robustness` в `config/settings.yaml`).

---

## Оценка и бенчмаркинг

- **Метрика соревнования**: MSE на тестовых данных (оценка организаторами) или на отложенной валидации локально.
- **Встроенная оценка**:
  - автоматический расчёт MSE на валидационной выборке после обучения;
  - сохранение результатов в «память экспериментов» для сравнения архитектур/гиперпараметров.
- **Benchmarking ML**: [`scripts/run_benchmark.py`](scripts/run_benchmark.py) — несколько моделей/гиперов на одном препроцессинге.
- **Сравнение сценариев системы** (RAG / без RAG и т.д.): [`scripts/compare_architectures.py`](scripts/compare_architectures.py) — два конфига подряд, метки в `experiments.jsonl`, копии submission.

---

## Стек технологий

- **LLM**: OpenRouter (OpenAI-совместимый клиент в [`src/llm/client.py`](src/llm/client.py)).
- **Оркестрация агентов**: явный Supervisor в Python (Coordinator), без LangGraph/AutoGen в репозитории.
- **RAG**: ChromaDB, эмбеддинги sentence-transformers (см. `src/rag/`).
- **Данные и ML**: pandas, scikit-learn, при необходимости LightGBM/XGBoost.
- **Конфиг**: YAML/ENV для API-ключей, путей к данным и к артефактам.

---

## Структура репозитория (рекомендуемая)

```
mws-ai-agents-2026/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.yaml              # основной конфиг (в т.ч. preprocessing_search, Optuna)
│   └── settings.norag.yaml      # вариант без RAG (compare_architectures)
├── data/
├── scripts/                       # вспомогательные сценарии (не вызываются из run.py)
│   ├── build_rag_index.py
│   ├── compare_architectures.py
│   ├── run_benchmark.py           # несколько моделей подряд (функция run_benchmark внутри)
│   └── run_pipeline_manual.py
├── src/
│   ├── agents/
│   │   ├── explorer.py
│   │   ├── engineer.py
│   │   ├── builder.py
│   │   ├── coordinator.py
│   │   └── tools_for_llm.py
│   ├── tools/
│   │   ├── eda_tools.py
│   │   ├── feature_tools.py
│   │   ├── model_tools.py
│   │   └── advanced_ensemble.py # стекинг / псевдо-лейблы (по конфигу evaluation)
│   ├── rag/
│   ├── memory/
│   │   └── experiments.py
│   ├── monitoring/
│   │   ├── logger.py
│   │   └── run_summary.py
│   ├── security/
│   │   ├── validation.py
│   │   ├── sanitize.py
│   │   └── guardrails.py
│   └── evaluation/
│       ├── hparam_search.py
│       ├── optuna_hparam.py
│       ├── preprocessing_search.py
│       ├── artifact_validation.py
│       └── code_validation.py
├── knowledge/
├── artifacts/
├── submissions/
└── run.py                         # единственная точка входа автоматического пайплайна
```

---

## Критерии курса — соответствие

Сводная таблица с **обоснованием решений**, ограничениями и ссылками на код — в **[docs/COMPLIANCE.md](docs/COMPLIANCE.md)**.

| Критерий | Реализация (кратко) |
|----------|---------------------|
| **Архитектура (20%)** | 3+ агента, Coordinator, RAG, паттерны; **feedback** по артефактам Engineer и по MSE/CV у Builder. |
| **Автоматизация и безопасность (20%)** | `run.py`, validation, guardrails, мониторинг токенов/инструментов. |
| **Документация (20%)** | README, ARCHITECTURE, RESULTS, COMPLIANCE, конфиг, `experiments.jsonl`. |
| **Качество модели (20%)** | MSE, CV, Optuna/сетка, ensemble, robustness (клип и др.). |
| **Benchmarking (20%)** | Бенчмарк ML, сравнение сценариев (RAG и др.), журнал экспериментов. |

---

## Задача соревнования (напоминание)

- **Метрика**: MSE.
- **Формат submission**: CSV с колонками `index`, `prediction` (непрерывные значения).
- Датасеты: `train.csv` (с `target`), `test.csv` (без `target`); все колонки, перечисленные в описании соревнования, должны учитываться в EDA и фичах.

Архитектура ориентирована на задачу регрессии с MSE в формате Kaggle-соревнования.
