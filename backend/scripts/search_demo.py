"""检索验证。

用法：python -m backend.scripts.search_demo "LangChain memory 模块的已知问题"
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK 控制台兼容

from ..app.config import settings
from ..app.rag.retriever import Retriever


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "LangChain memory module known issues and bugs"
    r = Retriever(repo=settings.target_repo)
    print(f"query: {q}\n")
    hits = r.search(q, limit=5)
    if not hits:
        print("无命中")
        return
    for h in hits:
        p = h["payload"]
        text = (h["text"] or "").replace("\n", " ")[:140]
        print(f"- [{p.get('source_type')} #{p.get('number')} | {p.get('status')} | {p.get('kind')}] {text}")


if __name__ == "__main__":
    main()
