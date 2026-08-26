"""CLI：对一个问题跑完整审计（planner → supervisor[三 worker 并行] → reflect → aggregator）。

用法（项目根目录，conda activate tdas）：
    python -m backend.scripts.audit "LangChain memory 模块的已知问题"
    python -m backend.scripts.audit "..." --json
"""
from __future__ import annotations

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ..app.agents.graph import build_graph
from ..app.config import settings
from ..app.db import DB
from ..app.llm import LLM
from ..app.rag.retriever import Retriever

DEFAULT_Q = "LangChain 的 memory 模块在生产环境里有哪些已知问题？"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_Q)
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON 报告")
    args = ap.parse_args()

    llm = LLM()
    db = DB()
    retriever = Retriever(repo=settings.target_repo)
    graph = build_graph(llm, retriever, db, settings.target_repo)

    result = graph.invoke({"question": args.question, "repo": settings.target_repo})
    report = result["report"]

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"## {report['question']}\n")
    print(f"计划: {report['plan']}")
    print(f"元数据: {report['metadata']}\n")
    for f in report["findings"]:
        print(f"### [{f['worker']}] {f['subtask']}")
        print(f"证据等级: {f['evidence_grade']} | 置信: {f['confidence']}")
        print(f["claim"])
        if f["sources"]:
            print("来源:")
            for s in f["sources"][:5]:
                snip = (s.get("snippet") or "")[:90].replace("\n", " ")
                if s.get("number") is not None:
                    ref = f"#{s.get('number')} ({s.get('type')}/{s.get('status')})"
                else:
                    ref = f"doc ({s.get('type')}) {s.get('url') or ''}"
                print(f"  - {ref} {snip}")
        print()

    # 反思：冲突清单
    conflicts = (report.get("reflection") or {}).get("conflicts", [])
    if conflicts:
        print("## ⚠️ 反思：结论冲突\n")
        for c in conflicts:
            print(f"- **{c.get('a')}**  ⟷  **{c.get('b')}**")
            print(f"  依据: {c.get('sources_a')} vs {c.get('sources_b')}")
            print(f"  矛盾点: {c.get('description')}")
            print(f"  裁决: {c.get('resolution')}\n")
    else:
        print("## 反思：未发现结论冲突\n")


if __name__ == "__main__":
    main()
