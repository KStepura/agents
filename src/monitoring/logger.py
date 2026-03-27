from __future__ import annotations

import logging
import sys
from typing import Any

_LOGGER = logging.getLogger("mws_agents.tools")


def configure_logging(level: int = logging.INFO) -> None:
    """Идемпотентная базовая конфигурация логирования для пакета агентов."""
    root = logging.getLogger("mws_agents")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def log_tool_call(agent: str, tool_name: str, duration_ms: float, **extra: Any) -> None:
    parts = [f"agent={agent}", f"tool={tool_name}", f"duration_ms={duration_ms:.1f}"]
    for k, v in extra.items():
        if v is not None:
            parts.append(f"{k}={v}")
    _LOGGER.info(" ".join(parts))
