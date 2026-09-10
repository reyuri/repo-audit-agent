"""Reflection：冲突识别 + 来源溯源 + 证据分级（D5 核心 agent 逻辑，跑强推理模型）。

审计类 agent 的关键一环：不只看「有没有证据」，还要看「证据是否互相矛盾、哪个更可信」。
分两层实现（可解释性是设计目标）：

1. 确定性层（规则、可审计）：
   - source_weight()：给每条来源按 (type, status, kind) 打可信度分。
     合并 PR(1.0) > 官方文档(0.8) > 未合并 PR(0.6) > 已关闭 issue(0.55)
     > 开放 issue(0.45) > 社区评论(0.35)。
   - grade_evidence()：据此给每个 worker 结论打 evidence_grade
     (strong / moderate / weak / insufficient)。
2. LLM 层（Pro）：把各 worker 结论（claim + 来源）喂给反思器，识别结论间矛盾，
   并对每个冲突给出「依据来源可信度更可信的一方」。

设计要点：这是「反思/自省」（reflection）机制——supervisor 调度 worker 查证后，
reflect 回头审视「查到的证据是否自洽」，把互相矛盾的发现显式标出来，而不是拼一份
看起来都对的报告。确定性分级保证可复现，LLM 只做语义矛盾判断。
"""
from __future__ import annotations

import json
import time

from ..config import settings
from ..db import DB


def source_weight(s: dict) -> float:
    """来源可信度分。分档依据：越接近「已落地的代码变更」越可信。"""
    st = (s.get("type") or s.get("source_type") or "").lower()
    status = (s.get("status") or "").upper()
    kind = (s.get("kind") or "").lower()
    if kind == "comment":
        return 0.35           # 社区单点声音，未经维护者确认
    if st == "pr":
        return 1.0 if status == "MERGED" else 0.6   # 已合并 > 未合并
    if st == "doc":
        return 0.8            # 官方文档权威，但可能滞后于代码
    if st == "issue":
        return 0.55 if status == "CLOSED" else 0.45  # 已关闭(真实 bug 报告) > 开放
    return 0.3


def grade_evidence(sources: list[dict]) -> str:
    """把一组来源汇总成一个证据等级。规则透明、可复现。"""
    if not sources:
        return "insufficient"
    ws = [source_weight(s) for s in sources]
    best = max(ws)
    n_high = sum(1 for w in ws if w >= 0.6)   # doc 或 PR 级
    n_any = sum(1 for w in ws if w >= 0.45)   # issue 级及以上
    if best >= 1.0 or n_high >= 2:
        return "strong"        # 有合并 PR 坐实，或多个 doc/PR 级交叉印证
    if best >= 0.6 or n_any >= 2:
        return "moderate"      # 单 doc / 未合并 PR，或多个 issue 级来源
    return "weak"              # 仅评论或单条开放 issue


REFLECT_PROMPT = """你是审计反思器。下面是多个 worker 对同一问题的审计结论（每条含关键断言与来源）。

任务：识别结论之间是否存在【矛盾或冲突】。典型例子：
- A 说「某 bug 已修复（PR #x 已合并）」，B 说「该 bug 在生产仍出现且无修复」。
- A 说「官方推荐用 X」，B 说「社区普遍反映 X 已弃用 / 不可用」。

只输出 JSON（不要其它文字）：
{{"conflicts": [
  {{"a": "结论1的关键断言", "b": "结论2的关键断言",
    "sources_a": ["#编号1"], "sources_b": ["#编号2"],
    "description": "矛盾点一句话",
    "resolution": "根据来源可信度（合并PR > 官方文档 > 已关闭issue > 评论），哪一方更可信及理由"}}
]}}
若没有实质矛盾，输出 {{"conflicts": []}}。"""


def _format_outputs(outputs: list[dict]) -> str:
    lines = []
    for i, o in enumerate(outputs, 1):
        srcs = ", ".join(sorted({
            f"#{s.get('number')}({s.get('type')}/{s.get('status')})"
            for s in o.get("sources", []) if s.get("number") is not None
        }))
        lines.append(
            f"[worker {i} / {o.get('worker')}] {o.get('query')}\n"
            f"  断言: {(o.get('answer') or '')[:200]}\n"
            f"  来源: {srcs or '(无)'}"
        )
    return "\n\n".join(lines)


def _parse_conflicts(text: str) -> list[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        obj = json.loads(text[start:end + 1])
        conflicts = obj.get("conflicts", [])
    except Exception:
        return []
    return [c for c in conflicts if isinstance(c, dict)][:5]


def detect_conflicts(question: str, outputs: list[dict], llm, db: DB, repo: str) -> list[dict]:
    """LLM（Pro）识别结论间矛盾。"""
    prompt = [
        {"role": "system", "content": REFLECT_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n{_format_outputs(outputs)}"},
    ]
    t0 = time.time()
    try:
        text = llm.chat(prompt, model=settings.llm_model_reasoning)
    except Exception as e:
        db.trace(agent="reflect", node="conflict", latency_ms=int((time.time() - t0) * 1000),
                 detail=f"LLM 调用失败: {str(e)[:300]}")
        return []
    db.trace(agent="reflect", node="conflict", latency_ms=int((time.time() - t0) * 1000),
             detail=text[:800])
    return _parse_conflicts(text)


def reflect_node(state: dict, llm, db: DB, repo: str) -> dict:
    outputs = state.get("worker_outputs", [])

    # 确定性：证据分级（不依赖 LLM，可复现）
    grades = [
        {"worker": o.get("worker"), "query": o.get("query"),
         "evidence_grade": grade_evidence(o.get("sources", []))}
        for o in outputs
    ]

    # LLM：结论间冲突识别（至少两条结论才有意义；全无来源则跳过省成本）
    conflicts: list[dict] = []
    if len(outputs) >= 2 and any(o.get("sources") for o in outputs):
        conflicts = detect_conflicts(state["question"], outputs, llm, db, repo)

    return {"reflection": {"grades": grades, "conflicts": conflicts}}
