"""Aggregator：合并各 worker 发现 → 结构化审计报告。

每个 finding 含：claim / sources / evidence_grade / confidence。
D4：按来源去重；metadata 记 unique_sources 数。
D5：消费 reflect 的 grades（证据分级）+ conflicts（冲突），置信度由证据等级推导。
"""
from __future__ import annotations

_GRADE_CONF = {"strong": "high", "moderate": "medium", "weak": "low", "insufficient": "low"}


def _src_key(s: dict) -> tuple:
    num = s.get("number")
    if num is None:
        return ("doc", (s.get("snippet") or "")[:80])
    return (num, s.get("type") or s.get("source_type"))


def _dedup_sources(sources: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for s in sources:
        k = _src_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def aggregator_node(state):
    question = state["question"]
    plan = state.get("plan", [])
    outputs = state.get("worker_outputs", [])
    seen_sources = state.get("seen_sources", [])
    reflection = state.get("reflection", {})
    grades = reflection.get("grades", [])

    # 证据等级按 worker_outputs 顺序对齐（reflect 与 aggregator 消费同一列表顺序）
    grade_by_idx = {i: g.get("evidence_grade", "insufficient") for i, g in enumerate(grades)}

    findings = []
    for i, o in enumerate(outputs):
        sources = _dedup_sources(o.get("sources", []))[:10]
        grade = grade_by_idx.get(i, "supported" if sources else "insufficient")
        findings.append({
            "worker": o.get("worker"),
            "subtask": o.get("query"),
            "claim": o.get("answer"),
            "sources": sources,
            "evidence_grade": grade,
            "confidence": _GRADE_CONF.get(grade, "low"),
        })

    report = {
        "question": question,
        "repo": state.get("repo"),
        "plan": plan,
        "findings": findings,
        "reflection": reflection,
        "metadata": {
            "worker_outputs": len(outputs),
            "total_steps": sum(o.get("steps", 0) for o in outputs),
            "covered_types": sorted({o.get("worker") for o in outputs if o.get("worker")}),
            "unique_sources": len(seen_sources),
            "conflicts": len(reflection.get("conflicts", [])),
        },
    }
    return {"findings": findings, "report": report}
