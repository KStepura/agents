"""
Проверка «кода» в артефактах Engineer: синтаксис .py в каталоге artifacts и
валидность сохранённого препроцессора (sklearn Pipeline в joblib).
Дополняет validate_processed_datasets в feedback loop Coordinator.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.security.validation import validate_path


def _iter_py_files(root: Path, *, max_depth: int = 4, max_files: int = 32) -> list[Path]:
    out: list[Path] = []
    root = root.resolve()

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth or len(out) >= max_files:
            return
        try:
            for p in sorted(d.iterdir()):
                if p.is_dir() and p.name not in (".git", "__pycache__", ".venv", "venv"):
                    walk(p, depth + 1)
                elif p.suffix == ".py":
                    out.append(p)
                    if len(out) >= max_files:
                        return
        except OSError:
            return

    walk(root, 0)
    return out


def validate_python_scripts_in_artifacts(
    artifacts_dir: str,
    allowed_dirs: list[str],
    *,
    max_depth: int = 4,
    max_files: int = 32,
) -> tuple[bool, list[str]]:
    """
    py_compile для каждого *.py под artifacts_dir (если такие файлы есть).
    При отсутствии .py возвращает (True, []).
    """
    errors: list[str] = []
    base = validate_path(str(artifacts_dir), allowed_dirs, must_exist=True)
    py_files = _iter_py_files(base, max_depth=max_depth, max_files=max_files)
    if not py_files:
        return True, []
    for fp in py_files:
        try:
            py_compile.compile(str(fp), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"Синтаксис Python: {fp.name}: {e.msg}")
        except OSError as e:
            errors.append(f"Python файл {fp}: {e}")
    return (len(errors) == 0, errors)


def validate_preprocessor_joblib(
    pipeline_path: str,
    allowed_dirs: list[str],
) -> tuple[bool, list[str]]:
    """
    Загружает joblib с препроцессором и проверяет наличие sklearn Pipeline
    (как в feature_tools: dict с ключом 'pipeline').
    """
    errors: list[str] = []
    try:
        p = validate_path(str(pipeline_path), allowed_dirs, must_exist=True)
    except (PermissionError, FileNotFoundError) as e:
        return False, [f"Препроцессор: {e}"]

    try:
        data = joblib.load(p)
    except Exception as e:
        return False, [f"Не удалось загрузить joblib препроцессора: {e}"]

    if isinstance(data, Pipeline):
        return True, []

    if isinstance(data, dict) and "pipeline" in data:
        pipe = data.get("pipeline")
        if isinstance(pipe, Pipeline):
            return True, []
        errors.append("В артефакте 'pipeline' не является sklearn.pipeline.Pipeline")
    else:
        errors.append("Ожидался dict с ключом 'pipeline' или sklearn Pipeline")

    return (len(errors) == 0, errors)
