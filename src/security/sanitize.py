"""
Ограничения на строковые аргументы инструментов LLM: длина, запрет null-байтов,
смягчение типичных попыток prompt-injection в полях, не предназначенных для инструкций.
"""

from __future__ import annotations

import re
from typing import Any

# Сообщения модели не должны попадать в пути/имена файлов как «инструкции»
_SUSPICIOUS_IN_PATH = re.compile(
    r"(?i)(ignore\s+(previous|all)\s+instructions|system\s*prompt|</?\s*script)",
)

MAX_TOOL_STRING = 500_000
MAX_PATH_STRING = 8_192


def sanitize_large_text(s: Any, *, field: str, max_len: int = MAX_TOOL_STRING) -> str:
    """Текстовые поля (EDA-отчёт и т.д.): ограничение размера и опасных байтов."""
    if s is None:
        return ""
    if not isinstance(s, str):
        raise TypeError(f"{field}: expected string")
    if "\x00" in s:
        raise ValueError(f"{field}: null bytes are not allowed")
    if len(s) > max_len:
        raise ValueError(f"{field}: length {len(s)} exceeds max {max_len}")
    return s


def sanitize_path_argument(s: Any, *, field: str, max_len: int = MAX_PATH_STRING) -> str:
    """Пути к файлам: компактные строки без управляющих инъекций."""
    if s is None or s == "":
        raise ValueError(f"{field}: path is required")
    if not isinstance(s, str):
        raise TypeError(f"{field}: expected string")
    if "\x00" in s or "\r" in s:
        raise ValueError(f"{field}: invalid characters in path")
    if len(s) > max_len:
        raise ValueError(f"{field}: path too long")
    if _SUSPICIOUS_IN_PATH.search(s):
        raise ValueError(f"{field}: path contains disallowed patterns")
    return s
