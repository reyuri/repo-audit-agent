"""分块 + 批量嵌入 + 入 Qdrant。**支持断点续传**：已入库的 point_id（SQLite doc_texts）自动跳过。

用法：
    python -m backend.scripts.build_index              # 全量索引（可断点续传）
    python -m backend.scripts.build_index --limit 500  # spike 抽样
    python -m backend.scripts.build_index --force      # 重建 collection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK 控制台兼容

from ..app.config import data_path, settings
from ..app.db import DB
from ..app.rag.indexer import Indexer, chunk_comment, chunk_doc, point_key

FLUSH = 2000  # 批量嵌入/写入大小


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _existing_point_ids(db: DB) -> set[str]:
    rows = db.conn.execute("SELECT point_id FROM doc_texts").fetchall()
    return {r[0] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    owner, repo = settings.target_repo.split("/")
    base = data_path("raw_data_dir") / f"{owner}__{repo}"
    items_path = base / "issues_prs.jsonl"
    comments_path = base / "comments.jsonl"

    idx = Indexer(repo=settings.target_repo)
    if args.force:
        idx.recreate()
    db = DB()
    existing = _existing_point_ids(db)
    print(f"[resume] 已有 {len(existing)} 个 point，跳过已索引内容", flush=True)

    def keep_missing(docs: list[dict]) -> list[dict]:
        """只保留还没入库的 chunk（确定性 uuid5 → 幂等续传）。"""
        out = []
        for d in docs:
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(d["meta"])))
            if pid not in existing:
                out.append(d)
        return out

    def flush(buf: list[dict]):
        if buf:
            idx.add_docs(buf)
            buf.clear()

    n = 0
    buf: list[dict] = []
    if os.path.exists(items_path):
        for rec in _iter_jsonl(str(items_path)):
            buf.extend(keep_missing(chunk_doc(rec)))
            if len(buf) >= FLUSH:
                flush(buf)
            n += 1
            if n % 1000 == 0:
                print(f"\r  body 记录 {n}", end="", flush=True)
            if args.limit and n >= args.limit:
                break
        flush(buf)
        print(f"\n✅ 正文处理完成（{n} 条记录）", flush=True)

    c = 0
    buf = []
    if os.path.exists(comments_path):
        for rec in _iter_jsonl(str(comments_path)):
            buf.extend(keep_missing([chunk_comment(rec)]))
            if len(buf) >= FLUSH:
                flush(buf)
            c += 1
            if c % 5000 == 0:
                print(f"\r  comments {c}", end="", flush=True)
        flush(buf)
        print(f"\n✅ 评论处理完成（{c} 条）", flush=True)

    total = idx.client.count(idx.collection).count
    print(f"collection 总点数：{total}", flush=True)


if __name__ == "__main__":
    main()
