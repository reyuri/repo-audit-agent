"""统一 LLM 客户端（OpenAI 兼容，可切换 provider）。

- chat：普通对话
- chat_with_tools：函数调用（ReAct 工具循环用）

模型分工（config 里配置，具体模型在私有 .env）：强推理模型做编排/反思，
低成本快速模型做检索 Worker。

可观测：方法级 `@traceable(name="llm.chat" / "llm.chat_with_tools", run_type="llm")`，
每次调用是一个 LLM span；原始 OpenAI 响应的 `r.usage` 被提取成 `usage_metadata`
放进 span 的 outputs，LangSmith 据此展示 token。配置了 LangSmith key（config.py 会写
env）时自动上报，无 key 时零开销降级。

线程注意：审计的 worker 跑在 ThreadPoolExecutor 子线程里，Python 默认不传播
contextvar，子线程里的 LLM 调用会丢父 trace 变孤儿。解法见 supervisor._run_parallel
—— 用 `contextvars.copy_context()` + `ctx.run(...)` 把节点 trace 上下文带进 worker 线程。
"""
from __future__ import annotations

import json

from langsmith import traceable
from openai import OpenAI

from .config import settings


def _usage_meta(u) -> dict:
    """OpenAI 响应 usage 对象 → LangSmith usage_metadata 结构。"""
    if not u:
        return {}
    return {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url or None,
            api_key=api_key or settings.llm_api_key,
        )

    # ---- 可观测内部：真实调用并返回 {content, tool_calls, usage_metadata}，
    #      让 @traceable 的 LLM span 带上 token（光返回字符串 capture 不到 usage）----

    @traceable(name="llm.chat", run_type="llm", metadata={"provider": "llm"})
    def _chat_once(self, messages: list[dict], model: str, temperature: float,
                   max_tokens: int) -> dict:
        r = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return {"content": r.choices[0].message.content or "",
                "usage_metadata": _usage_meta(r.usage)}

    @traceable(name="llm.chat_with_tools", run_type="llm", metadata={"provider": "llm"})
    def _tools_once(self, messages: list[dict], tools: list[dict], model: str,
                    temperature: float, max_tokens: int) -> dict:
        r = self.client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=temperature, max_tokens=max_tokens,
        )
        msg = r.choices[0].message
        calls: list[dict] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return {"content": msg.content or "", "tool_calls": calls,
                "usage_metadata": _usage_meta(r.usage)}

    # ---- 公有接口：只给调用方返回需要的值 ----

    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        # max_tokens 给足：推理模型的 reasoning_content 与 content 共享预算，
        # 太小会被思考吃光导致 content 为空
        out = self._chat_once(messages, model or settings.llm_model_fast, temperature, max_tokens)
        return out["content"]

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        model: str | None = None, temperature: float = 0.2,
                        max_tokens: int = 2048) -> tuple[str, list[dict]]:
        """返回 (content, tool_calls)。tool_calls: [{id, name, args}]。"""
        out = self._tools_once(messages, tools, model or settings.llm_model_fast,
                               temperature, max_tokens)
        return out["content"], out["tool_calls"]
