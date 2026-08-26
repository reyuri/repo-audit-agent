"""分块 + 入 Qdrant 嵌入式（QdrantLocal，零 Docker）。

设计：Qdrant payload **只存元数据**（过滤用），**全文存 SQLite（doc_texts）**按
point_id 关联。原因：QdrantLocal 写大 payload 很慢（实测 ~129 pts/s），分离后
写入快数倍，且全文按需取回——向量库管检索、关系库管文本。

分块按源结构：issue/PR = 标题+正文，超长按段落切（max_chars=2000，避免
MiniLM 512 token 截断浪费计算）；评论单独成块。点 id 用确定性 uuid5 → 幂等。
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from ..config import data_path, settings
from ..db import DB
from . import collection_for
from .embedder import Embedder

EMBED_DIM = settings.embed_dim  # all-MiniLM-L6-v2=384；bge-base-en-v1.5=768


def _split_text(text: str, max_chars: int = 2000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > max_chars and cur:
            parts.append(cur)
            cur = ""
        cur += para + "\n\n"
    if cur:
        parts.append(cur)
    return parts or [""]


def _meta(rec: dict, source_type: str, kind: str, part: int = 0, **extra) -> dict:
    m = {
        "source_type": source_type,
        "number": rec.get("number"),
        "date": rec.get("created_at"),
        "status": rec.get("state"),
        "author": rec.get("user"),
        "labels": rec.get("labels", []),
        "url": rec.get("url"),
        "kind": kind,
        "part": part,
    }
    m.update(extra)
    return m


def chunk_doc(rec: dict) -> list[dict]:
    """issue/PR 记录 → 分块文档列表。"""
    source_type = "issue" if rec.get("type") == "issue" else "pr"
    head = f"[{source_type.upper()}] {rec.get('title', '')}".strip()
    body = (rec.get("body") or "").strip()
    if not body:
        return [{"text": head or "(no body)", "meta": _meta(rec, source_type, "body")}]
    return [
        {"text": f"{head}\n\n{seg}", "meta": _meta(rec, source_type, "body", part=i)}
        for i, seg in enumerate(_split_text(body))
    ]


def chunk_comment(rec: dict) -> dict:
    """评论记录 → 单块文档。"""
    body = (rec.get("body") or "").strip() or "(empty comment)"
    return {
        "text": f"[COMMENT on #{rec.get('number')} ({rec.get('source_type')})] {body}",
        "meta": _meta(
            {"number": rec.get("number"), "created_at": rec.get("created_at"),
             "state": rec.get("status"), "user": rec.get("author"), "url": rec.get("url")},
            rec.get("source_type") or "issue", "comment",
            comment_id=rec.get("comment_id"),
        ),
    }


def point_key(meta: dict) -> str:
    if meta.get("source_type") == "doc":
        # 文档用 path 保证跨文件唯一（number 为 None 会撞）
        return f"doc_{meta.get('path')}_{meta.get('part', 0)}"
    return f"{meta.get('source_type')}_{meta.get('number')}_{meta.get('kind')}_{meta.get('comment_id', meta.get('part'))}"


def _split_markdown_sections(text: str, max_chars: int = 2000) -> list[tuple[str, str]]:
    """官方文档分块：按 markdown 标题（# 开头）切段，超长段再按段落切。"""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []

    def flush(heading: str, body: list[str]):
        body_text = "\n".join(body).strip()
        if not body_text and not heading:
            return
        if len(body_text) <= max_chars:
            sections.append((heading, body_text))
        else:
            for seg in _split_text(body_text, max_chars):
                sections.append((heading, seg))

    cur_heading, cur_body = "", []
    for line in lines:
        if line.startswith("#"):
            flush(cur_heading, cur_body)
            cur_heading = line
            cur_body = []
        else:
            cur_body.append(line)
    flush(cur_heading, cur_body)
    return sections


def chunk_doc_text(text: str, path: str, repo: str = settings.target_repo,
                   branch: str = "main") -> list[dict]:
    """官方文档 → 分块文档列表（按 markdown 标题切）。"""
    title = path.split("/")[-1]
    docs = []
    for i, (heading, body) in enumerate(_split_markdown_sections(text)):
        docs.append({
            "text": f"[DOC] {heading or title}\n\n{body}",
            "meta": {
                "source_type": "doc",
                "number": None,
                "date": None,
                "status": None,
                "author": None,
                "labels": [],
                "url": f"https://github.com/{repo}/blob/{branch}/{path}",
                "kind": "doc",
                "path": path,
                "part": i,
            },
        })
    return docs


class Indexer:
    def __init__(self, repo: str = settings.target_repo, collection: str | None = None):
        self.repo = repo
        self.collection = collection or collection_for(repo)
        self.client = QdrantClient(path=str(data_path("qdrant_path")))
        self.embedder = Embedder.get()
        self.db = DB()
        self._ensure_collection()

    def _ensure_collection(self):
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            # 维度不匹配（换模型）时自动重建，避免 768/384 冲突
            try:
                info = self.client.get_collection(self.collection)
                cur = info.config.params.vectors.size
                if cur != EMBED_DIM:
                    print(f"[indexer] 维度 {cur} != {EMBED_DIM}，重建 collection", flush=True)
                    self.client.delete_collection(self.collection)
                    names.discard(self.collection)
            except Exception:
                pass
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        for f, schema in (("source_type", PayloadSchemaType.KEYWORD),
                          ("status", PayloadSchemaType.KEYWORD),
                          ("number", PayloadSchemaType.INTEGER),
                          ("kind", PayloadSchemaType.KEYWORD)):
            try:
                self.client.create_payload_index(self.collection, field_name=f, field_schema=schema)
            except Exception:
                pass  # 已存在则忽略；本地模式本就不支持，无害

    def recreate(self):
        try:
            self.client.delete_collection(self.collection)
        except Exception as e:
            print(f"[warn] delete_collection: {e}", flush=True)
        # 重新建
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            names.discard(self.collection)
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def add_docs(self, docs: list[dict], wait: bool = False) -> int:
        if not docs:
            return 0
        texts = [d["text"] for d in docs]
        vectors = self.embedder.embed(texts)
        points = []
        text_map: dict[str, str] = {}
        for d, vec, text in zip(docs, vectors, texts):
            m = d["meta"]
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key(m)))
            payload = {"repo": self.repo, **m, "text_id": pid}  # 不含全文
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
            text_map[pid] = text
        self.client.upsert(collection_name=self.collection, points=points, wait=wait)
        self.db.put_doc_texts(text_map)
        return len(points)
