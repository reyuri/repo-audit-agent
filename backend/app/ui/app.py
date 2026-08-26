"""TDAS — 技术文档深度审计系统 · Streamlit UI。

启动（项目根目录，conda activate tdas）：
    streamlit run backend/app/ui/app.py

亮点：用 graph.stream(stream_mode="updates") 实时展示各 agent 节点（planner →
supervisor → reflect → aggregator）的完成进度；报告里带证据分级、冲突反思、
agent_traces 可观测时间线 —— 面试时「可观测的多 Agent」就是这么呈现的。
"""
from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# streamlit run 的脚本加载不保证包上下文（且 CLI/AppTest 行为不同），
# 这里显式把项目根塞进 sys.path，用绝对导入 —— 自包含，任何 CWD 都能跑。
_ROOT = pathlib.Path(__file__).resolve().parents[3]  # backend/app/ui -> 项目根
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from backend.app.agents.graph import build_graph
from backend.app.rag.retriever import Retriever
from backend.app.config import settings
from backend.app.db import DB
from backend.app.llm import LLM

st.set_page_config(page_title="TDAS · 技术文档深度审计", page_icon="🔍", layout="wide")

DEFAULT_Q = "LangChain 的 memory 模块在生产环境里有哪些已知 bug、性能陷阱和已修复的问题？"

GRADE_COLORS = {
    "strong": "#16a34a",
    "moderate": "#d97706",
    "weak": "#dc2626",
    "insufficient": "#6b7280",
    "supported": "#16a34a",
}
GRADE_LABELS = {
    "strong": "证据充分（合并PR/多方印证）",
    "moderate": "证据中等（单文档/未合并PR）",
    "weak": "证据薄弱（评论/开放issue）",
    "insufficient": "证据不足",
}


def _fmt_badge(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#6b7280")
    label = GRADE_LABELS.get(grade, grade)
    return f'<span style="background:{color}22;color:{color};border:1px solid {color};border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600">{label}</span>'


def _source_link(s: dict, repo: str) -> str:
    n = s.get("number")
    t = (s.get("type") or s.get("source_type") or "")
    if n is not None:
        path = "pull" if t == "pr" else "issues"
        return f"https://github.com/{repo}/{path}/{n}"
    return s.get("url") or ""


@st.cache_resource
def _get_llm():
    return LLM()


@st.cache_resource
def _get_db():
    return DB()


@st.cache_resource
def _get_retriever():
    return Retriever(repo=settings.target_repo)


def render_overview(report: dict):
    meta = report["metadata"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worker 产出", meta["worker_outputs"])
    c2.metric("工具调用步数", meta["total_steps"])
    c3.metric("覆盖类型", ", ".join(meta["covered_types"]))
    c4.metric("冲突数", meta["conflicts"])
    st.caption(f"独立来源数 {meta['unique_sources']} · 仓库 {report['repo']}")
    st.divider()


def render_plan(plan: list[dict]):
    st.subheader("📋 审计计划（Planner 分解）")
    for t in plan:
        icon = {"doc": "📄", "issue": "🐞", "code": "🔧"}.get(t.get("type"), "•")
        st.markdown(f"- {icon} **{t.get('type')}** — `{t.get('query')}`")


def render_findings(findings: list[dict], repo: str):
    st.subheader(f"🔎 审计发现（{len(findings)} 条）")
    for f in findings:
        grade = f.get("evidence_grade") or f.get("evidence_status") or "insufficient"
        with st.expander(
            f"[{f.get('worker')}] {f.get('subtask')}",
            expanded=grade in ("strong", "moderate"),
        ):
            st.markdown(f"{_fmt_badge(grade)} · 置信 **{f.get('confidence')}**",
                        unsafe_allow_html=True)
            st.markdown(f["claim"])
            if f.get("sources"):
                st.markdown("**来源：**")
                for s in f["sources"][:6]:
                    ref = f"#{s.get('number')}" if s.get("number") is not None else "doc"
                    url = _source_link(s, repo)
                    tip = f"*{ref} ({s.get('type')}/{s.get('status')})* "
                    snip = (s.get("snippet") or "")[:120].replace("\n", " ")
                    st.markdown(f"- {tip}[{snip}]({url})" if url else f"- {tip}{snip}")
            else:
                st.warning("无来源 —— 该结论未查证到证据")


def render_reflection(report: dict):
    reflection = report.get("reflection") or {}
    conflicts = reflection.get("conflicts") or []
    grades = reflection.get("grades") or []
    st.subheader("🧠 反思层（Reflect）")
    if grades:
        for i in range(0, len(grades), 4):  # 每行最多 4 个，worker 多时换行
            c = st.columns(min(4, len(grades) - i))
            for col, g in zip(c, grades[i:i + 4]):
                col.markdown(
                    f"{_fmt_badge(g['evidence_grade'])}<br><small>{g['worker']}</small>",
                    unsafe_allow_html=True,
                )
    if conflicts:
        st.warning(f"检出 {len(conflicts)} 个结论冲突：")
        for conf in conflicts:
            st.markdown(f"- **{conf.get('a')}** ⟷ **{conf.get('b')}**")
            st.markdown(f"  - 矛盾点：{conf.get('description')}")
            st.markdown(f"  - 裁决：{conf.get('resolution')}")
    else:
        st.success("未检出结论冲突 —— 各 worker 发现互补、证据自洽")


def render_traces(db: DB, trace_from: int):
    rows = db.conn.execute(
        "SELECT agent, node, tool, COUNT(*), AVG(latency_ms), SUM(CASE WHEN node='act' THEN 1 ELSE 0 END) "
        "FROM agent_traces WHERE id > ? GROUP BY agent, node "
        "ORDER BY agent, node", (trace_from,)
    ).fetchall()
    if not rows:
        return
    st.subheader("👁️ Agent 轨迹（可观测性）")
    st.caption("每次 planner 规划 / worker 工具调用 / supervisor 决策 / reflect 反思都落库，可复盘「谁调了什么、花了多久」")
    df = [{"agent": r[0], "node": r[1], "tool": r[2] or "-", "次数": r[3],
           "平均耗时(ms)": round(r[4] or 0, 1), "工具调用": r[5]} for r in rows]
    st.dataframe(df, use_container_width=True, hide_index=True)


def run_audit(question: str, llm, db, retriever, repo: str) -> dict:
    graph = build_graph(llm, retriever, db, repo)
    status = st.status("多 Agent 审计进行中…", expanded=True)
    report = None
    with status:
        for chunk in graph.stream({"question": question, "repo": repo}, stream_mode="updates"):
            for node, upd in chunk.items():
                if node == "planner":
                    plan = upd.get("plan") or []
                    status.update(label=f"✅ Planner：拆解出 {len(plan)} 个子任务")
                    st.markdown("**Planner** 将问题分解为子任务：")
                    for t in plan:
                        st.markdown(f"  - `{t.get('type')}` {t.get('query')}")
                elif node == "supervisor":
                    n = len(upd.get("worker_outputs") or [])
                    status.update(label=f"🔄 Supervisor：已并行产出 {n} 个 worker 结论（含动态追问）")
                elif node == "reflect":
                    nc = len((upd.get("reflection") or {}).get("conflicts") or [])
                    status.update(label=f"🧠 Reflect：证据分级完成，检出 {nc} 个冲突")
                elif node == "aggregator":
                    status.update(label="✅ Aggregator：审计报告生成")
                    report = upd.get("report")
        status.update(label="✅ 审计完成", state="complete")
    return report


def main():
    st.title("🔍 TDAS — 多 Agent 技术文档深度审计系统")
    st.caption("LangGraph supervisor 编排 · Qdrant+BM25 混合检索(RRF) · 反思层冲突识别")

    with st.sidebar:
        st.header("⚙️ 控制台")
        repo = settings.target_repo
        st.markdown(f"**目标仓库** `{repo}`")

        retriever = _get_retriever()
        db = _get_db()
        try:
            n_points = retriever.client.count(retriever.collection).count
        except Exception:
            n_points = 0
        try:
            n_texts = db.conn.execute("SELECT COUNT(*) FROM doc_texts").fetchone()[0]
        except Exception:
            n_texts = 0
        st.markdown(f"**数据底座**：\n- Qdrant 向量点 `{n_points:,}`\n- 全文文本 `{n_texts:,}`")
        st.divider()
        st.caption("模型分工：Supervisor/Planner/Reflect → Pro · 检索 Worker → Flash")

    question = st.text_area("要审计的问题", value=DEFAULT_Q, height=90)
    run = st.button("🚀 运行审计", type="primary", use_container_width=True)

    if run:
        llm = _get_llm()
        trace_from = db.conn.execute("SELECT COALESCE(MAX(id),0) FROM agent_traces").fetchone()[0]
        with st.spinner("加载检索与模型…"):
            report = run_audit(question, llm, db, retriever, repo)
        if report is None:
            st.error("审计失败：未生成报告（看 agent_traces 排查）")
        else:
            st.session_state["report"] = report
            st.session_state["trace_from"] = trace_from

    # 渲染结果（跑完直接显示；或展示上次结果）
    report = st.session_state.get("report")
    trace_from = st.session_state.get("trace_from", 0)
    if report:
        st.divider()
        st.header(f"📊 审计报告：{report['question']}")
        render_plan(report.get("plan") or [])
        st.divider()
        render_overview(report)
        render_reflection(report)
        render_findings(report["findings"], report.get("repo") or repo)
        render_traces(db, trace_from)
    else:
        st.info("输入问题后点「运行审计」。示例问题覆盖三类 worker：文档 / issue / PR。")


if __name__ == "__main__":
    main()
