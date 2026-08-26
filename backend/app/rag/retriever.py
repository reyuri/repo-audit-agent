"""统一检索入口：BM25 + ANN 混合，RRF 融合，支持元数据硬过滤。

返回 [{pid, payload, text, score}]。
- ANN：Qdrant 向量检索（payload 只存元数据，全文从 SQLite doc_texts 取）。
- BM25：rank_bm25，语料与向量索引同源（同一 chunk + 确定性 uuid5），可 pickle 缓存。
- RRF：结果在 point_id 上融合（1/(k+rank)），兼顾语义与关键词。

usage:
    r = Retriever()
    hits = r.search("LangChain memory 已知问题", limit=10, source_type="issue")
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from ..config import data_path, settings
from ..db import DB
from . import collection_for
from .bm25 import BM25Index
from .embedder import Embedder

_POOL = 60     # 混合检索各自召回池大小（再 RRF 取前 limit）
_ANN_POOL = 300  # ANN 无过滤召回池：先大池召回再 Python 过滤，避免 QdrantLocal 线性扫描过滤

# 低信号评论（Dosu / 自动化 triage / GitHub actions）：数量巨大且文本泛化，
# 会同时污染 BM25 与 ANN 把真实结果挤掉。查询时过滤（不重索引，快速见效）。
BOT_AUTHORS = {
    "dosubot", "langchain-oss-automated-triage", "github-actions[bot]",
    "github-actions", "dosu",
}
BOT_MARKERS = ("_🤖_", "I'm an AI assistant", "manage their backlog", "I'm Dosu")


def build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conds = []
    for k, v in filters.items():
        if k == "number":
            conds.append(FieldCondition(key=k, range=Range(gte=v, lte=v)))
        elif isinstance(v, (list, tuple)):
            conds.append(FieldCondition(key=k, match=MatchValue(value=v[0])))  # 简化：取首个
        else:
            conds.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=conds)


def rrf_fuse(ann: list[dict], bm25: list[dict], k: int | None = None, limit: int = 10) -> list[str]:
    """RRF 融合：在 point_id 上合并两个排序列表。返回排序后的 pid 列表。"""
    k = k or settings.rrf_k
    acc: dict[str, float] = {}
    for items in (ann, bm25):
        for rank, item in enumerate(items):
            pid = item["pid"]
            acc[pid] = acc.get(pid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(acc.items(), key=lambda x: -x[1])
    return [pid for pid, _ in ordered[:limit]]


class Retriever:
    def __init__(self, repo: str = settings.target_repo, collection: str | None = None):
        self.repo = repo
        self.collection = collection or collection_for(repo)
        self.client = QdrantClient(path=str(data_path("qdrant_path")))
        self.embedder = Embedder.get()
        self.db = DB()
        self._bm25: BM25Index | None = None
        self._num_idx: dict[int, list[str]] | None = None  # number -> [pid]，懒构建

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            try:
                self._bm25 = BM25Index.load(self.repo)
            except FileNotFoundError:
                raise RuntimeError("BM25 缓存未构建，请先运行 python -m backend.scripts.build_bm25")
        return self._bm25

    # ---------- 内部 ----------
    def _ann(self, query: str, limit: int, filters: dict | None, source_type: str | None) -> list[dict]:
        """ANN 检索。关键优化：**不把 source_type/status 传给 Qdrant**——
        QdrantLocal 带 filter 的 ANN 是线性扫描（实测 ~4.5s），无过滤走 HNSW（~0.9s）。
        改为无过滤大召回池（_ANN_POOL）+ Python 侧硬过滤，快 ~5× 且结果等价。
        """
        vec = self.embedder.embed([query])[0]
        try:
            res = self.client.query_points(
                collection_name=self.collection, query=vec,
                limit=_ANN_POOL, with_payload=True,
            )
            points = res.points
        except AttributeError:  # 旧版 qdrant-client
            res = self.client.search(
                collection_name=self.collection, query_vector=vec,
                limit=_ANN_POOL, with_payload=True,
            )
            points = res
        # Python 侧硬过滤（source_type + 其它元数据），QdrantLocal 里做会线性扫描
        out = []
        for p in points:
            payload = p.payload or {}
            if source_type and payload.get("source_type") != source_type:
                continue
            if filters:
                skip = False
                for k, v in filters.items():
                    if k == "source_type":
                        continue
                    if payload.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            out.append(p)
            if len(out) >= limit:
                break
        ids = [str(p.id) for p in out]
        texts = self.db.get_doc_texts(ids)
        return [
            {"pid": str(p.id), "payload": p.payload, "text": texts.get(str(p.id), ""),
             "score": float(getattr(p, "score", 0.0))}
            for p in out
        ]

    def _fetch_details(self, pids: list[str]) -> list[dict]:
        """按 pid 取回 payload（Qdrant）与全文（SQLite）。"""
        texts = self.db.get_doc_texts(pids)
        payload_map: dict[str, dict] = {}
        try:
            res = self.client.retrieve(self.collection, ids=pids, with_payload=True)
            payload_map = {str(r.id): r.payload for r in res}
        except Exception:
            pass
        return [
            {"pid": pid, "payload": payload_map.get(pid, {}), "text": texts.get(pid, ""), "score": None}
            for pid in pids
        ]

    def _number_index(self) -> dict[int, list[str]]:
        """懒构建 number -> point_id 索引（与 BM25 同源），避免 Qdrant 全库扫描。"""
        if self._num_idx is None:
            idx: dict[int, list[str]] = {}
            for pid, m in self.bm25.meta.items():
                n = m.get("number")
                if n is not None:
                    idx.setdefault(n, []).append(pid)
            self._num_idx = idx
        return self._num_idx

    @staticmethod
    def _is_bot(item: dict) -> bool:
        """判断一条命中是否是机器人低信号评论。"""
        author = ((item.get("payload") or {}).get("author") or "").lower()
        if author in BOT_AUTHORS:
            return True
        text = (item.get("text") or "")[:300].lower()
        return any(m in text for m in BOT_MARKERS)

    def _is_bot_pid(self, pid: str) -> bool:
        author = ((self.bm25.meta.get(pid) or {}).get("author") or "").lower()
        return author in BOT_AUTHORS

    def get_by_number(self, number: int, limit: int = 60) -> list[dict]:
        """按 issue/PR 编号取全部 chunk。

        关键优化：不用 Qdrant scroll 过滤（QdrantLocal 会对全库线性扫描，实测 ~4s/次），
        改为 BM25 meta 内存索引按 number 查 point_id，再按 id 点查 —— O(1)。
        """
        pids = self._number_index().get(number, [])[:limit]
        details = self._fetch_details(pids)
        return [{"payload": d["payload"], "text": d["text"], "score": None} for d in details]

    # ---------- 对外 ----------
    def warmup(self):
        """单线程预热 BM25 与 number 索引。

        关键：并行 worker 首次访问 self.bm25 会并发触发 309MB pickle 懒加载，
        多线程同时 load 互相阻塞（实测检索被拖到 14-19s/次）。派发 worker 前先预热。
        """
        _ = self.bm25
        _ = self._number_index()

    def search(self, query: str, limit: int | None = None, filters: dict | None = None,
               source_type: str | None = None, hybrid: bool = True) -> list[dict]:
        """统一检索。hybrid=True 走 BM25+ANN+RRF；False 走纯 ANN。"""
        limit = limit or settings.top_k
        ann = [a for a in self._ann(query, _POOL, filters, source_type) if not self._is_bot(a)]
        if not hybrid:
            return ann[:limit]
        bm25 = [b for b in self.bm25.search(query, limit=_POOL, source_type=source_type, filters=filters)
                if not self._is_bot_pid(b["pid"])]
        fused = rrf_fuse(ann, bm25, limit=limit)
        details = self._fetch_details(fused)
        # 尽量带 ANN 分数便于调试
        score_map = {a["pid"]: a["score"] for a in ann}
        for d in details:
            d["score"] = score_map.get(d["pid"], d["score"])
        return details
