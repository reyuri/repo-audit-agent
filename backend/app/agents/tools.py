"""工具集：search_docs / search_issues / search_prs / get_issue_detail。

工具 schema 为 OpenAI function-calling 格式，实现层都走 Retriever（混合检索）。

D4 新增：按 worker 类型分发工具（不同 worker 只能调用自己职责内的工具）——
- doc_worker：search_docs
- issue_worker：search_issues + get_issue_detail
- code_worker：search_prs + search_issues + get_issue_detail（通过 PR diff/讨论获取代码级证据）
代码符号级检索（AST/定义引用）为未来工作，当前 code worker 的「代码证据」来自 PR diff。
"""
from __future__ import annotations

from ..rag.retriever import Retriever

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "在官方文档中检索技术主题（语义+关键词混合）。返回文档片段与来源。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要检索的技术问题/关键词"},
            "limit": {"type": "integer", "description": "返回条数（默认 5）"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_issues",
        "description": "在 GitHub issue 讨论中检索技术问题。可用 status 过滤。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要检索的问题/关键词"},
            "status": {"type": "string", "enum": ["open", "closed"], "description": "可选：按 issue 状态过滤"},
            "limit": {"type": "integer", "description": "返回条数（默认 5）"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_prs",
        "description": "在 GitHub PR 及其 diff/讨论中检索代码级证据（修复、重构、行为变更）。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要检索的代码关键词/改动主题"},
            "status": {"type": "string", "enum": ["open", "closed", "merged"], "description": "可选：按 PR 状态过滤"},
            "limit": {"type": "integer", "description": "返回条数（默认 5）"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_issue_detail",
        "description": "按编号获取某个 issue/PR 的完整正文与评论（纯元数据过滤）。",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer", "description": "issue/PR 编号"},
        }, "required": ["number"]}}},
]


def _hit_to_tool(h: dict) -> dict:
    p = h["payload"]
    return {
        "number": p.get("number"),
        "type": p.get("source_type"),
        "kind": p.get("kind"),
        "status": p.get("status"),
        "author": p.get("author"),
        "url": p.get("url"),
        "score": round(float(h.get("score") or 0), 3),
        "snippet": (h.get("text") or "")[:300],
    }


def _detail_to_tool(h: dict) -> dict:
    p = h["payload"]
    return {
        "number": p.get("number"),
        "type": p.get("source_type"),
        "kind": p.get("kind"),
        "author": p.get("author"),
        "text": (h.get("text") or "")[:2000],
    }


def build_tools(retriever: Retriever) -> list[dict]:
    """构建全部工具（doc/issue/code 三类都含）。graph 里用 build_tools_for 按 worker 取子集。"""
    def search_docs(query: str, limit: int = 5):
        return [_hit_to_tool(h) for h in retriever.search(query, limit=limit, source_type="doc")]

    def search_issues(query: str, status: str | None = None, limit: int = 5):
        filters = {"status": status} if status else None
        hits = retriever.search(query, limit=limit, source_type="issue", filters=filters)
        return [_hit_to_tool(h) for h in hits]

    def search_prs(query: str, status: str | None = None, limit: int = 5):
        filters = {"status": status} if status else None
        hits = retriever.search(query, limit=limit, source_type="pr", filters=filters)
        return [_hit_to_tool(h) for h in hits]

    def get_issue_detail(number: int):
        hits = retriever.get_by_number(number)
        return [_detail_to_tool(h) for h in hits]

    fns = {
        "search_docs": search_docs, "search_issues": search_issues,
        "search_prs": search_prs, "get_issue_detail": get_issue_detail,
    }
    return [{"name": t["function"]["name"], "schema": t, "fn": fns[t["function"]["name"]]}
            for t in TOOL_SCHEMAS]


# worker 类型 → 允许调用的工具名（职责边界：每个 worker 只能看到自己的工具）
WORKER_TOOLS: dict[str, list[str]] = {
    "doc": ["search_docs"],
    "issue": ["search_issues", "get_issue_detail"],
    "code": ["search_prs", "search_issues", "get_issue_detail"],
}


def build_tools_for(retriever: Retriever, worker_type: str) -> list[dict]:
    """按 worker 类型取工具子集（D4 的「按 worker 分发工具」）。"""
    allowed = WORKER_TOOLS.get(worker_type, ["search_docs"])
    return [t for t in build_tools(retriever) if t["name"] in allowed]
