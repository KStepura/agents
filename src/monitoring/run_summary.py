"""
Сводка прогона для мониторинга: длительность, метрики, использование LLM.
Пишется в artifacts/run_summary.json (включение в config/settings.yaml).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_summary(
    artifacts_dir: str,
    payload: dict[str, Any],
) -> str | None:
    """Сохраняет JSON. Возвращает путь к файлу или None при ошибке."""
    out = Path(artifacts_dir) / "run_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts_end": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    try:
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError:
        return None
    return str(out)
