"""Worker：手写 ReAct 工具循环（Reason→Act→Observe），有界 MAX_TOOL_STEPS。

每个 worker 内部保留透明可观测的工具调用循环——这是项目亮点，不让多 agent 层
吞掉「function calling 可观测」这个卖点。每步 act 都落 agent_traces 表。

D4：worker 支持 seen_sources 提示（短期 Memory 去重），提示本 worker 避开已引用的
来源，去挖新的证据，避免多个 worker 反复引用同一 issue/PR 造成报告冗余。
"""
from __future__ import annotations

import json
import time

from ..config import settings
from ..db import DB
from .state import MAX_STEPS_BY_MODE, MODE_DEEP

WORKER_PROMPT = """你是 {worker_type} 研究员，负责审计开源项目 {repo}。

子任务：{query}

规则：
1. 只能使用提供的工具查证，不要凭空编造。
2. 每次工具调用后观察结果，决定继续查证还是给出结论。
3. 结论必须基于工具返回的来源，用来源编号引用（如 [来源1]）。
4. 查不到证据就明确说「证据不足」，绝不硬编。
5. 最多调用 {max_steps} 次工具。
{seen_hint}
"""


def _seen_hint(seen_sources: list[dict]) -> str:
    """把已引用来源压缩成一句提示，避免重复查证。"""
    if not seen_sources:
        return ""
    nums = sorted({f"#{s.get('number')}" for s in seen_sources if s.get("number")})
    if not nums:
        return ""
    shown = ", ".join(nums[:15]) + ("…" if len(nums) > 15 else "")
    return f"6. 以下来源已被其它 worker 引用，优先挖新的证据、避免重复：{shown}。"


def run_react(query: str, worker_type: str, tools: list[dict], llm, db: DB, repo: str,
              seen_sources: list[dict] | None = None, max_steps: int | None = None) -> dict:
    """跑一个 ReAct 循环。max_steps 按审计模式（fast/deep）传入，默认 deep。"""
    max_steps = max_steps or MAX_STEPS_BY_MODE[MODE_DEEP]
    tool_map = {t["name"]: t["fn"] for t in tools}
    tool_schemas = [t["schema"] for t in tools]
    messages = [
        {"role": "system", "content": WORKER_PROMPT.format(
            worker_type=worker_type, repo=repo, query=query, max_steps=max_steps,
            seen_hint=_seen_hint(seen_sources or []))},
        {"role": "user", "content": query},
    ]
    sources: list[dict] = []
    final_answer: str | None = None
    used_steps = 0

    for step in range(1, max_steps + 1):
        used_steps = step
        t0 = time.time()
        try:
            content, calls = llm.chat_with_tools(messages, tool_schemas,
                                                 model=settings.llm_model_fast)
        except Exception as e:
            db.trace(agent=worker_type, node="reason", latency_ms=int((time.time() - t0) * 1000),
                     detail=f"LLM 调用失败: {str(e)[:300]}")
            final_answer = f"[LLM 调用失败，无法完成查证] {str(e)[:200]}"
            break

        if not calls:
            final_answer = content or "(worker 未给出结论)"
            break

        # 记录 assistant 的工具调用，回填到消息
        messages.append({
            "role": "assistant", "content": content,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in calls
            ],
        })
        for c in calls:
            tt = time.time()
            try:
                result = tool_map.get(c["name"], lambda **_: {"error": "未知工具"})(**c["args"])
            except Exception as e:
                result = {"error": str(e)[:200]}
            db.trace(agent=worker_type, node="act", tool=c["name"],
                     latency_ms=int((time.time() - tt) * 1000),
                     detail=json.dumps(result, ensure_ascii=False)[:800])
            # 记录来源（供最终引用）
            if isinstance(result, list):
                for h in result[:5]:
                    sources.append({"tool": c["name"], "query": c["args"].get("query"), **h})
            messages.append({
                "role": "tool", "tool_call_id": c["id"],
                "content": json.dumps(result, ensure_ascii=False)[:4000],
            })

    if final_answer is None:
        # 达到最大工具步数仍未自然收敛：强制收敛——基于已查证来源做最后一次无工具总结，
        # 避免「查了一堆证据却没结论」的坏输出。
        messages.append({
            "role": "user",
            "content": "已到达工具调用上限。请基于以上已查证的来源，用简洁的结论回答原问题，"
                       "标明证据来源；若证据不足请明确说明「证据不足」。不要再调用工具。",
        })
        try:
            final_answer = llm.chat(messages, model=settings.llm_model_fast) or "(无结论)"
        except Exception as e:
            final_answer = f"[强制收敛失败] {str(e)[:200]}"

    return {"worker": worker_type, "query": query, "answer": final_answer,
            "sources": sources, "steps": used_steps}
