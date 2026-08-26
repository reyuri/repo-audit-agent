"""构建 BM25 索引并缓存（data/cache/bm25_<repo>.pkl）。

用法：python -m backend.scripts.build_bm25
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ..app.config import settings
from ..app.rag.bm25 import BM25Index


def main():
    bm = BM25Index(repo=settings.target_repo)
    bm.save()
    print(f"✅ BM25 索引构建完成：{len(bm.docs)} chunks -> "
          f"data/cache/bm25_{settings.target_repo.replace('/', '__')}.pkl")


if __name__ == "__main__":
    main()
