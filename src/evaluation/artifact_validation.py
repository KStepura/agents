"""
Проверка артефактов Engineer (обработанные train/test) перед этапом подбора модели.
Используется Coordinator для feedback loop: при ошибках — повторный вызов Engineer с текстом замечаний.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_processed_datasets(
    train_path: str,
    test_path: str,
    *,
    target_col: str = "target",
    min_train_rows: int = 10,
    min_test_rows: int = 1,
) -> tuple[bool, list[str]]:
    """
    Возвращает (успех, список сообщений об ошибках).
    Не открывает файлы вне переданных путей (вызывать с уже проверенными путями).
    """
    errors: list[str] = []
    tp = Path(train_path)
    sp = Path(test_path)
    if not tp.exists():
        errors.append(f"Нет файла train_processed: {train_path}")
        return False, errors
    if not sp.exists():
        errors.append(f"Нет файла test_processed: {test_path}")
        return False, errors

    try:
        tr = pd.read_csv(tp)
        te = pd.read_csv(sp)
    except Exception as e:
        errors.append(f"Ошибка чтения CSV: {e}")
        return False, errors

    if tr.shape[0] < min_train_rows:
        errors.append(f"Слишком мало строк в train ({tr.shape[0]}), нужно >= {min_train_rows}")
    if te.shape[0] < min_test_rows:
        errors.append(f"Слишком мало строк в test ({te.shape[0]})")

    if target_col not in tr.columns:
        errors.append(f"В train нет колонки цели '{target_col}'")

    tr_feats = [c for c in tr.columns if c != target_col]
    te_feats = [c for c in te.columns if c != target_col]
    set_tr = set(tr_feats)
    set_te = set(te_feats)
    if set_tr != set_te:
        only_tr = set_tr - set_te
        only_te = set_te - set_tr
        if only_tr:
            errors.append(f"Колонки только в train (должны быть и в test): {sorted(only_tr)[:10]}...")
        if only_te:
            errors.append(f"Колонки только в test (должны совпадать с train): {sorted(only_te)[:10]}...")

    if tr_feats:
        na_all = [c for c in tr_feats if tr[c].isna().all()]
        if na_all:
            errors.append(f"Колонки из одних NaN в train: {na_all[:8]}")

    return (len(errors) == 0, errors)
