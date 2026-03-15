"""
OpenRouter-compatible LLM client and agent loop with tool calling.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def get_client(llm_config: dict) -> "OpenAI":
    """Create OpenAI-compatible client for OpenRouter."""
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
) -> tuple[str, list[dict]]:
    """
    Run chat with tool use loop. Returns (final assistant text, full messages).
    tool_executor(name, arguments_dict) -> result string.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tools = [{"type": "function", "function": f} for f in tools_schema]
    step = 0
    final_text = ""

    while step < max_steps:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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

    return final_text, messages
