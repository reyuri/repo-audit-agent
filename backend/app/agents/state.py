"""LangGraph 共享状态。"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    repo: str
    plan: list[dict]           # planner 输出: [{id, type: doc|issue|code, query}]
    worker_outputs: list[dict]  # 各 worker 的原始输出（ReAct answer + sources）
    seen_sources: list[dict]    # 短期 Memory：本轮已引用的来源 [{number, source_type, kind}]
    reflection: dict           # reflect 输出: {grades: [...], conflicts: [...]}
    findings: list[dict]        # aggregator 输出: [{claim, sources, evidence_grade, ...}]
    report: dict                # 最终结构化审计报告
