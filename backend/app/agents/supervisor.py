"""Supervisor：动态调度器（D4 核心 agent 逻辑，跑强推理模型）。

D3 是静态线性流 planner→doc_worker→aggregator；D4 引入 supervisor 做「动态调度」：

1. 第 1 轮：并行派发 planner 分解出的全部子任务。doc/issue/code 三类 worker 各带
   自己的工具子集（build_tools_for），用 ThreadPoolExecutor 并行跑。
2. 第 2..N 轮：supervisor 用 Pro 模型读当前已产出（各 worker 结论 + 已引用来源），
   判断证据是否充分；不足则动态追加追问子任务（dynamic re-planning），继续并行派发。
3. 短期 Memory 去重：seen_sources 记录已引用 (number, source_type)，派发前注入 worker
   提示避开（见 worker._seen_hint），聚合前再由 aggregator 按来源去重。

这是「supervisor + workers」编排模式（类似 AutoGen group chat / CrewAI crew）：
supervisor 只做调度与证据充分性判断，不亲自检索；worker 只做本职责内的查证。
"""
from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import settings
from ..db import DB
from .state import MAX_FOLLOWUPS_BY_MODE, MAX_ROUNDS_BY_MODE, MAX_STEPS_BY_MODE, MODE_DEEP
from .tools import WORKER_TOOLS, build_tools_for
from .worker import run_react

SUPERVISOR_PROMPT = """你是审计 supervisor，负责判断当前证据是否充分，必要时追加查证。

用户问题：{question}

已完成的子任务与结论：
{completed}

已引用的来源编号（避免重复查证）：{seen}

判断：是否还有关键盲区需要追加查证？例如某 worker 提到的 issue 引出了更深的 PR 修复、
某类证据（文档/issue/代码）缺失、或结论之间出现矛盾需要进一步求证。

只输出 JSON，不要任何其它文字：
{{"followups": [{{"type": "doc|issue|code", "query": "具体追问（英文技术关键词）"}}]}}

若证据已充分、无需追加，输出 {{"followups": []}}。最多 {max_followups} 条。"""


def _dedup_key(s: dict) -> tuple:
    """来源去重键：issue/PR 按 (number, source_type)；doc 无 number，用 snippet 前缀。"""
    num = s.get("number")
    if num is None:
        return ("doc", (s.get("snippet") or "")[:80])
    return (num, s.get("type") or s.get("source_type"))


def _update_seen(seen: list[dict], output: dict) -> None:
    """把一次 worker 输出的来源并入短期 Memory（按去重键去重）。"""
    existing = {_dedup_key(s) for s in seen}
    for s in output.get("sources", []):
        k = _dedup_key(s)
        if k not in existing:
            existing.add(k)
            seen.append({
                "number": s.get("number"),
                "source_type": s.get("type") or s.get("source_type"),
                "kind": s.get("kind"),
            })


def _run_parallel(tasks: list[dict], retriever, llm, db: DB, repo: str,
                  seen: list[dict], max_steps: int) -> list[dict]:
    """并行跑一组子任务（每个 task 按其 type 走对应 worker + 工具子集）。"""
    if not tasks:
        return []
    tools_by_type = {wt: build_tools_for(retriever, wt) for wt in WORKER_TOOLS}

    def work(t: dict) -> dict:
        wt = t.get("type") if t.get("type") in WORKER_TOOLS else "doc"
        return run_react(t.get("query", ""), wt, tools_by_type[wt], llm, db, repo, seen, max_steps)

    # 把当前节点（supervisor）的 LangSmith trace 上下文带进 worker 线程：
    # ThreadPoolExecutor 默认不传播 contextvar，子线程里的 LLM 调用会丢父 trace
    # 变孤儿（token 也挂不到节点树）。copy_context + ctx.run 复现父上下文。
    # 注意 Context 对象不可跨线程共享，必须每个 worker 单独 copy 一份。
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as ex:
        futures = []
        for t in tasks:
            child_ctx = contextvars.copy_context()
            futures.append(ex.submit(lambda t=t, c=child_ctx: c.run(work, t)))
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:  # 单 worker 崩溃不拖垮整个审计
                results.append({"worker": "?", "query": "?",
                                "answer": f"[worker 异常] {str(e)[:200]}",
                                "sources": [], "steps": 0})
    return results


def _normalize_followups(fu, max_followups: int) -> list[dict]:
    out = []
    for f in (fu or []):
        if isinstance(f, dict) and f.get("type") in WORKER_TOOLS and f.get("query"):
            out.append({"id": None, "type": f["type"], "query": f["query"].strip()[:200]})
    return out[:max_followups]


def _parse_followups(text: str, max_followups: int) -> list[dict]:
    """解析 supervisor 输出。支持 {"followups": [...]} 与纯数组两种形态，容错其它噪声。"""
    text = text.strip()
    # 优先：{"followups": [...]}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("followups"), list):
                return _normalize_followups(obj["followups"], max_followups)
        except Exception:
            pass
    # 兜底：纯数组
    start2, end2 = text.find("["), text.rfind("]")
    if start2 != -1 and end2 != -1:
        try:
            arr = json.loads(text[start2:end2 + 1])
            if isinstance(arr, list):
                return _normalize_followups(arr, max_followups)
        except Exception:
            pass
    return []


def _decide_followups(question: str, outputs: list[dict], seen: list[dict],
                      llm, db: DB, repo: str, rnd: int, max_followups: int) -> list[dict]:
    """supervisor 决策：读当前产出，决定是否追加追问（Pro 模型）。"""
    # 上下文压缩：Pro 调用耗时与输入长度强相关，只喂最近几条 + 截断答案
    completed = "\n".join(
        f"- [{o.get('worker')}] {o.get('query')}\n  {(o.get('answer') or '')[:200]}"
        for o in outputs[-8:]
    ) or "(无)"
    seen_str = ", ".join(sorted({f"#{s['number']}" for s in seen if s.get("number")})) or "(无)"
    prompt = [
        {"role": "system", "content": SUPERVISOR_PROMPT.format(
            question=question, completed=completed, seen=seen_str, max_followups=max_followups)},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    try:
        text = llm.chat(prompt, model=settings.llm_model_reasoning)
    except Exception as e:
        db.trace(agent="supervisor", node="decide", latency_ms=int((time.time() - t0) * 1000),
                 detail=f"LLM 调用失败: {str(e)[:300]}")
        return []
    db.trace(agent="supervisor", node="decide", latency_ms=int((time.time() - t0) * 1000),
             detail=f"round={rnd} {text[:800]}")
    return _parse_followups(text, max_followups)


def supervisor_node(state: dict, llm, retriever, db: DB, repo: str) -> dict:
    question = state["question"]
    plan = state.get("plan", [])
    mode = state.get("mode", MODE_DEEP)
    max_steps = MAX_STEPS_BY_MODE.get(mode, MAX_STEPS_BY_MODE[MODE_DEEP])
    max_rounds = MAX_ROUNDS_BY_MODE.get(mode, MAX_ROUNDS_BY_MODE[MODE_DEEP])
    max_followups = MAX_FOLLOWUPS_BY_MODE.get(mode, MAX_FOLLOWUPS_BY_MODE[MODE_DEEP])
    outputs = list(state.get("worker_outputs", []))
    seen = list(state.get("seen_sources", []))

    # 预热：BM25 + number 索引必须在并行 worker 前单线程加载（懒加载竞态会拖慢检索 ~14-19s/次）
    try:
        retriever.warmup()
    except Exception:
        pass  # 预热失败则 worker 内各自懒加载（慢但可用）

    # 第 1 轮：并行派发 planner 静态子任务（三类 worker 各干各的）
    tasks = [t for t in plan if t.get("type") in WORKER_TOOLS]
    if tasks:
        for r in _run_parallel(tasks, retriever, llm, db, repo, seen, max_steps):
            outputs.append(r)
            _update_seen(seen, r)

    # 第 2..N 轮：动态追加追问（supervisor 判断证据充分性，轮数按模式）
    for rnd in range(2, max_rounds + 1):
        followups = _decide_followups(question, outputs, seen, llm, db, repo, rnd, max_followups)
        if not followups:
            break
        db.trace(agent="supervisor", node="dispatch", detail=f"round={rnd} 追加 {len(followups)} 个子任务")
        for r in _run_parallel(followups, retriever, llm, db, repo, seen, max_steps):
            outputs.append(r)
            _update_seen(seen, r)

    return {"worker_outputs": outputs, "seen_sources": seen}
