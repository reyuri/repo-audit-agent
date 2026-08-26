"""统一 LLM 客户端（OpenAI 兼容，可切换 provider）。

- chat：普通对话
- chat_with_tools：函数调用（ReAct 工具循环用）

模型分工（config 里配置，具体模型在私有 .env）：强推理模型做编排/反思，
低成本快速模型做检索 Worker。

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
            base_url=base_url or settings.llm_base_url or None,
            api_key=api_key or settings.llm_api_key,
        )

    @traceable(name="llm.chat", run_type="llm", metadata={"provider": "llm"})
    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        # max_tokens 给足：推理模型的 reasoning_content 与 content 共享预算，
        # 太小会被思考吃光导致 content 为空
        r = self.client.chat.completions.create(
            model=model or settings.llm_model_fast,
            messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    @traceable(name="llm.chat_with_tools", run_type="llm", metadata={"provider": "llm"})
    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        model: str | None = None, temperature: float = 0.2,
                        max_tokens: int = 2048) -> tuple[str, list[dict]]:
        """返回 (content, tool_calls)。tool_calls: [{id, name, args}]。"""
        r = self.client.chat.completions.create(
            model=model or settings.llm_model_fast,
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
