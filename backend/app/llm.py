"""统一 LLM 客户端（DeepSeek 为主，OpenAI 兼容，可切 provider）。

- chat：普通对话
- chat_with_tools：函数调用（ReAct 工具循环用）

模型分工（config 里配置）：Flash 做检索 Worker，Pro 做 Planner/Supervisor/Reflect。

可观测：chat/chat_with_tools 都挂 @traceable（langsmith）——配置了 LangSmith key
时自动成为父 trace 的 LLM span（图级由 LangGraph 自动追踪）；无 key 时零开销降级。
"""
from __future__ import annotations

import json

from langsmith import traceable
from openai import OpenAI

from .config import settings


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(
            base_url=base_url or settings.deepseek_base_url,
            api_key=api_key or settings.deepseek_api_key,
        )

    @traceable(name="llm.chat", run_type="llm", metadata={"provider": "deepseek"})
    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        # max_tokens 给足：DeepSeek 推理模型的 reasoning_content 与 content 共享预算，
        # 太小会被思考吃光导致 content 为空（brief 第 7.2 条）
        r = self.client.chat.completions.create(
            model=model or settings.deepseek_model_flash,
            messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    @traceable(name="llm.chat_with_tools", run_type="llm", metadata={"provider": "deepseek"})
    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        model: str | None = None, temperature: float = 0.2,
                        max_tokens: int = 2048) -> tuple[str, list[dict]]:
        """返回 (content, tool_calls)。tool_calls: [{id, name, args}]。"""
        r = self.client.chat.completions.create(
            model=model or settings.deepseek_model_flash,
            messages=messages, tools=tools, tool_choice="auto",
            temperature=temperature, max_tokens=max_tokens,
        )
        msg = r.choices[0].message
        content = msg.content or ""
        calls: list[dict] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return content, calls
