"""eval_reflect.py：反思层（D5）评估集 —— 已知 ground truth 的冲突/自洽样本。

反思层的核心卖点是「抓出结论间矛盾」。为验证它，用一组**已知 ground truth** 的
样本：人为构造「互相矛盾」与「互补/自洽」的 worker 结论对，跑 reflect 的两层
（证据分级 + 冲突识别），比对期望结果打 PASS/FAIL。

用法（项目根目录，conda activate tdas）：
    python -m backend.scripts.eval_reflect            # 合成评估集（快，每例 1 次 Pro 调用）
    python -m backend.scripts.eval_reflect "问题"     # 完整流水线（planner→supervisor→reflect）

说明：合成样本只测 reflect 层（不含检索/worker），是「已知冲突评估集」的严格形式；
完整流水线见 audit.py。
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ..app.db import DB
from ..app.llm import LLM
from ..app.agents.reflect import detect_conflicts, grade_evidence

# (名称, 结论对, 期望是否检出冲突)
SYNTH_CASES = [
    (
        "矛盾：限制「可靠」 vs 「失效」",
        [
            {"worker": "doc", "query": "limit guarantee",
             "answer": "官方文档声称 ConversationSummaryBufferMemory 的 max_token_limit 能可靠地把注入 prompt 的历史压缩到上限以内。",
             "sources": [{"number": None, "type": "doc", "kind": "doc", "status": None}]},
            {"worker": "issue", "query": "limit failure",
             "answer": "issue #17888 报告 moving_summary_buffer 摘要不断膨胀，总 token 数突破 max_token_limit，限制失效。",
             "sources": [{"number": 17888, "type": "issue", "kind": "body", "status": "CLOSED"}]},
        ],
        True,
    ),
    (
        "矛盾：bug「已修复」 vs 「仍存在」",
        [
            {"worker": "code", "query": "fix landed",
             "answer": "内存泄漏已在 PR #32365 修复（register_configure_hook 只在模块导入时注册一次）。",
             "sources": [{"number": 32365, "type": "pr", "kind": "body", "status": "MERGED"}]},
            {"worker": "issue", "query": "still leaking",
             "answer": "多个生产用户仍报告该内存泄漏，问题尚未解决。",
             "sources": [{"number": 32300, "type": "issue", "kind": "body", "status": "OPEN"}]},
        ],
        True,
    ),
    (
        "自洽：两个 worker 结论互补、不冲突",
        [
            {"worker": "doc", "query": "best practice",
             "answer": "官方建议用 trim/summarize 控制上下文窗口。",
             "sources": [{"number": None, "type": "doc", "kind": "doc", "status": None}]},
            {"worker": "issue", "query": "common bugs",
             "answer": "ConversationBufferMemory 存在无限增长问题，社区有相关 bug 报告。",
             "sources": [{"number": 17888, "type": "issue", "kind": "body", "status": "CLOSED"}]},
        ],
        False,
    ),
]


def run_synthetic(llm, db):
    passed = 0
    for name, outputs, expect_conflict in SYNTH_CASES:
        grades = [grade_evidence(o["sources"]) for o in outputs]
        conflicts = detect_conflicts("（合成样本）", outputs, llm, db, "repo")
        got = len(conflicts) > 0
        ok = (got == expect_conflict)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        print(f"      证据分级={grades} | 期望冲突={expect_conflict} 实际冲突={got}")
        for c in conflicts:
            print(f"      裁决: {c.get('resolution')}")
    print(f"\n{passed}/{len(SYNTH_CASES)} 通过")


def run_e2e(question, llm, db):
    from ..app.agents.graph import build_graph
    from ..app.config import settings
    from ..app.rag.retriever import Retriever
    retriever = Retriever(repo=settings.target_repo)
    graph = build_graph(llm, retriever, db, settings.target_repo)
    report = graph.invoke({"question": question, "repo": settings.target_repo})["report"]
    reflection = report.get("reflection", {})
    print(f"worker 数={report['metadata']['worker_outputs']} | "
          f"证据等级={[g['evidence_grade'] for g in reflection.get('grades', [])]} | "
          f"冲突数={len(reflection.get('conflicts', []))}")
    for c in reflection.get("conflicts", []):
        print(f"  - {c.get('a')}  ⟷  {c.get('b')}")
        print(f"    裁决: {c.get('resolution')}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=None)
    args = ap.parse_args()

    llm = LLM()
    db = DB()
    if args.question:
        run_e2e(args.question, llm, db)
    else:
        run_synthetic(llm, db)


if __name__ == "__main__":
    main()
