"""
Клиент LLM, совместимый с OpenRouter, и агентный цикл с вызовом инструментов.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_client(llm_config: dict) -> "OpenAI":
    """Создаёт клиент, совместимый с OpenAI API, для работы с OpenRouter."""
    if OpenAI is None:
        raise ImportError("Install openai: pip install openai")
    api_key = os.environ.get(
        llm_config.get("api_key_env", "OPENROUTER_API_KEY"),
        llm_config.get("api_key", ""),
    )
    if not api_key:
        raise ValueError(
            f"Set {llm_config.get('api_key_env', 'OPENROUTER_API_KEY')} in .env or config"
        )
    return OpenAI(
        base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_api_calls": 0,
    }


def _merge_usage(acc: dict[str, int], response: Any) -> None:
    u = getattr(response, "usage", None)
    if u is None:
        return
    acc["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
    acc["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
    acc["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)
    acc["llm_api_calls"] += 1


def run_chat_with_tools(
    client: "OpenAI",
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    tools_schema: list[dict],
    tool_executor: Callable[[str, dict], str],
    max_steps: int = 15,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[str, list[dict], dict[str, int]]:
    """
    Запускает чат с циклом использования инструментов.
    Возвращает (финальный ответ ассистента, полный список сообщений, usage).

    usage: prompt_tokens, completion_tokens, total_tokens, llm_api_calls.
    tool_executor(name, arguments_dict) -> строка результата.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tools = [{"type": "function", "function": f} for f in tools_schema]
    step = 0
    final_text = ""
    usage = _empty_usage()

    while step < max_steps:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        _merge_usage(usage, response)
        choice = response.choices[0]
        msg = choice.message
        if msg.content:
            final_text = (msg.content or "").strip()
        if not getattr(msg, "tool_calls", None) or len(msg.tool_calls) == 0:
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = tool_executor(name, args)
            except Exception as e:
                result = f"Error: {e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)[:8000],
                }
            )
        step += 1

    return final_text, messages, usage
