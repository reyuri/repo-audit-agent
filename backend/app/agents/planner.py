"""Planner：把开放问题分解成结构化子任务 [{id, type: doc|issue|code, query}]。"""
from __future__ import annotations

import json
import time

from ..config import settings

PLANNER_PROMPT = """你是任务规划器。把用户对开源项目 {repo} 的技术审计问题，分解成 1-3 个可执行的子任务。

每个子任务必须是以下三种类型之一：
- doc：查官方文档（语义检索）
- issue：查 GitHub issue/PR 讨论（关键词 + 状态过滤）
- code：查代码 / PR diff（原型期能力有限）

只输出 JSON 数组，不要任何其它文字：
[{{"id": 1, "type": "doc", "query": "具体检索问句"}}, ...]

query 用英文技术关键词，便于检索命中。
"""


def parse_subtasks(text: str) -> list[dict]:
    """从 LLM 输出里提取 JSON 数组；失败时退化为单个 doc 子任务。"""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return [{"id": 1, "type": "doc", "query": text.strip()[:200]}]
    try:
        arr = json.loads(text[start:end + 1])
        return [s for s in arr if s.get("type") in ("doc", "issue", "code")][:3]
    except Exception:
        return [{"id": 1, "type": "doc", "query": text.strip()[:200]}]


def planner_node(state, llm, db, repo: str):
    prompt = [
        {"role": "system", "content": PLANNER_PROMPT.format(repo=repo)},
        {"role": "user", "content": state["question"]},
    ]
    t0 = time.time()
    try:
        text = llm.chat(prompt, model=settings.deepseek_model_pro)
    except Exception as e:
        db.trace(agent="planner", node="plan", latency_ms=int((time.time() - t0) * 1000),
                 detail=f"LLM 调用失败: {str(e)[:300]}")
        text = f'[{{"id":1,"type":"doc","query":"{state["question"][:100]}"}}]'
    db.trace(agent="planner", node="plan", latency_ms=int((time.time() - t0) * 1000),
             detail=text[:800])
    return {"plan": parse_subtasks(text)}
