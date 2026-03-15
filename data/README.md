# Данные

Файлы соревнования лежат в этой папке:

- `train.csv` — обучающая выборка (с колонкой `target`)
- `test.csv` — тестовая выборка (без `target`)
- `sample_submition.csv` — пример submission в формате `index,prediction` (в названии опечатка: submission)
- `solution.csv` — эталонный/референсный файл (если есть)

Каталог `data/` задаётся в `config/settings.yaml` как `paths.data_dir`.
