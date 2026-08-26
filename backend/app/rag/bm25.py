"""BM25 关键词检索（rank_bm25），供 ANN 混合经 RRF 融合。

关键点：BM25 语料 = 与向量索引**完全一致**的 chunk（同一 chunk_doc/chunk_comment +
确定性 uuid5），这样 BM25 命中与 ANN 命中能在 point_id 上对齐融合。
构建一次后 pickle 缓存到 data/cache/bm25_<repo>.pkl。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import uuid
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..config import data_path, settings
from .indexer import chunk_comment, chunk_doc, point_key


def tokenize(text: str) -> list[str]:
    # 英文关键词检索：小写 + 按非字母数字切分，保留下划线/连字符（代码符号友好）
    return re.findall(r"[a-z0-9][a-z0-9_\-]*", text.lower())


def _build_chunks(repo: str) -> list[dict]:
    """从原始 JSONL 重建 chunk 列表（与 build_index 同源同 chunk 函数 → 同 point_id）。"""
    owner, name = repo.split("/")
    base = data_path("raw_data_dir") / f"{owner}__{name}"
    docs: list[dict] = []

    items_path = base / "issues_prs.jsonl"
    if os.path.exists(items_path):
        with open(items_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for d in chunk_doc(rec):
                    d["pid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(d["meta"])))
                    docs.append(d)

    comments_path = base / "comments.jsonl"
    if os.path.exists(comments_path):
        with open(comments_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = chunk_comment(rec)
                d["pid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(d["meta"])))
                docs.append(d)

    # 官方文档（来自独立 docs 仓库，docs.jsonl 已是 {text, meta} chunk 格式）
    docs_owner, docs_name = settings.docs_repo.split("/")
    docs_base = data_path("raw_data_dir") / f"{docs_owner}__{docs_name}"
    docs_path = docs_base / "docs.jsonl"
    if os.path.exists(docs_path):
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d["pid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(d["meta"])))
                docs.append(d)
    return docs


class BM25Index:
    def __init__(self, repo: str = settings.target_repo, docs: list[dict] | None = None):
        self.repo = repo
        self.docs = docs if docs is not None else _build_chunks(repo)
        self.tokenized = [tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)
        self.meta = {d["pid"]: d["meta"] for d in self.docs}

    def search(self, query: str, limit: int = 20, source_type: str | None = None,
               filters: dict | None = None) -> list[dict]:
        """BM25 检索（可选元数据硬过滤）。返回 [{pid, score}]。"""
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = []
        for d, s in zip(self.docs, scores):
            m = d["meta"]
            if source_type and m.get("source_type") != source_type:
                continue
            if filters:
                skip = False
                for k, v in filters.items():
                    if k in ("source_type", "number"):
                        continue
                    if m.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            ranked.append((d["pid"], float(s)))
        ranked.sort(key=lambda x: -x[1])
        return [{"pid": pid, "score": s} for pid, s in ranked[:limit]]

    def save(self, path: str | Path | None = None):
        path = path or data_path("cache_dir") / f"bm25_{self.repo.replace('/', '__')}.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"docs": self.docs, "tokenized": self.tokenized}, f)

    @classmethod
    def load(cls, repo: str = settings.target_repo, path: str | Path | None = None) -> "BM25Index":
        path = path or data_path("cache_dir") / f"bm25_{repo.replace('/', '__')}.pkl"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 cache not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls.__new__(cls)
        inst.repo = repo
        inst.docs = data["docs"]
        inst.tokenized = data["tokenized"]
        inst.bm25 = BM25Okapi(inst.tokenized)
        inst.meta = {d["pid"]: d["meta"] for d in inst.docs}
        return inst
