"""LangGraph 共享状态 + 审计模式配置。"""
from __future__ import annotations

from typing import TypedDict

# ---- 审计模式：deep（深度，慢） / fast（快速） ----
MODE_DEEP = "deep"
MODE_FAST = "fast"
# 按模式定 worker 最大工具步数（步数与耗时线性相关，每步一次 LLM 调用）
MAX_STEPS_BY_MODE = {MODE_DEEP: 5, MODE_FAST: 3}
# 按模式定 supervisor 调度轮数 / 每轮最大追问数
MAX_ROUNDS_BY_MODE = {MODE_DEEP: 3, MODE_FAST: 2}
MAX_FOLLOWUPS_BY_MODE = {MODE_DEEP: 2, MODE_FAST: 1}


class AgentState(TypedDict, total=False):
    question: str
    repo: str
    mode: str                  # "deep" | "fast"（决定 worker 步数 + supervisor 轮数）
    plan: list[dict]           # planner 输出: [{id, type: doc|issue|code, query}]
    worker_outputs: list[dict]  # 各 worker 的原始输出（ReAct answer + sources）
    seen_sources: list[dict]    # 短期 Memory：本轮已引用的来源 [{number, source_type, kind}]
    reflection: dict           # reflect 输出: {grades: [...], conflicts: [...]}
    findings: list[dict]        # aggregator 输出: [{claim, sources, evidence_grade, ...}]
    report: dict                # 最终结构化审计报告
