"""抓取官方文档（docs/ 下的 markdown）并索引为 source_type=doc。

流程：git tree 列表 → raw 拉原文 → 按 markdown 标题分块 → 写 docs.jsonl → 入 Qdrant（幂等可续）。

用法（项目根目录，conda activate tdas）：
    python -m backend.scripts.ingest_docs               # 全量
    python -m backend.scripts.ingest_docs --limit 200   # 抽样
"""
from __future__ import annotations

import argparse
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ..app.config import data_path, settings
from ..app.db import DB
from ..app.ingest.docs_ingest import fetch_all_docs, save_docs_jsonl
from ..app.rag.indexer import Indexer, point_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（spike 用）")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    owner, repo = settings.docs_repo.split("/")  # 文档来自独立 docs 仓库

    # 1) 拉原文
    records, branch = fetch_all_docs(owner, repo, limit=args.limit, concurrency=args.concurrency)
    if not records:
        print("⚠️ 没抓到任何文档，检查 prefix（默认 docs/）与文件后缀")
        return

    # 2) 分块存 JSONL（BM25 复用）
    out_path, n_chunks = save_docs_jsonl(records, owner, repo)
    print(f"✅ 文档分块 {n_chunks} 条 -> {out_path}")

    # 3) 入 Qdrant（断点续传：已存在的 point_id 跳过）
    idx = Indexer(repo=settings.target_repo)
    db = DB()
    existing = {r[0] for r in db.conn.execute("SELECT point_id FROM doc_texts").fetchall()}
    import json as _json

    buf = []
    flushed = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(d["meta"])))
            if pid in existing:
                continue
            buf.append(d)
            if len(buf) >= 2000:
                idx.add_docs(buf)
                flushed += len(buf)
                buf = []
    if buf:
        idx.add_docs(buf)
        flushed += len(buf)
    print(f"✅ 新索引 {flushed} 个文档 chunk（collection {idx.collection}）")


if __name__ == "__main__":
    main()
