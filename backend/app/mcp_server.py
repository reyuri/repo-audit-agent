"""TDAS 检索层 MCP server（方向 B：把 14.3 万点 LangChain 索引暴露给任何 MCP 客户端）。

用法（项目根目录）：
    python -m backend.app.mcp_server           # stdio transport，等 MCP client 接入

接入 Claude Desktop / Cursor / 任何 MCP client，命令配置为：
    command = python, args = ["-m", "backend.app.mcp_server"], cwd = 项目根目录

设计：
- 与多 Agent 审计共用同一套 Retriever（BM25+ANN+RRF 混合检索）→ 一个能力两种身份。
- **主线程预热 + 常驻服务**：启动时主线程加载模型（~1min），然后常驻提供 stdio 或
  SSE/HTTP 服务。Claude Code 等客户端用 **SSE 连接常驻服务**（无启动超时问题）。
- 工具输出 JSON 字符串（MCP 工具返回字符串是惯例）。

⚠️ Windows 排雷（都实测踩过）：
1. **stdout 专跑 JSON-RPC 绝不能被污染**；stderr 不能向管道打 UTF-8（tqdm █/emoji）——
   Windows 客户端以 GBK 读 stderr 会解码崩溃/管道阻塞。已把 stderr 重定向到日志文件。
2. **任何非主线程首次 import torch/sklearn/scipy（sentence-transformers 依赖）都会与
   asyncio 事件循环死锁**（AnyIO worker 线程和独立后台线程都实测挂死）。所以预热必须在
   mcp.run 之前的主线程完成。这也是必须用常驻服务的原因（启动慢，客户端拉起必超时）。
3. **show_banner=False**：fastmcp 的 ASCII banner 含 UTF-8 表情，同样会被 GBK 客户端读崩。
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import warnings

# ⚠️ MCP stdio 协议：stdout 专跑 JSON-RPC，绝不能被污染。stderr 也不能向管道打 UTF-8
# （tqdm 加载条用 █、emoji 等），Windows 客户端以 GBK 读 stderr 会解码崩溃/管道阻塞
# → 客户端挂死。所以把 stderr 重定向到日志文件：协议通道最干净，日志可追溯。
_LOGDIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "logs"
_LOGDIR.mkdir(parents=True, exist_ok=True)
sys.stderr = open(_LOGDIR / "mcp_server.log", "a", encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")  # 已知 Qdrant 本地模式 UserWarning 等

from fastmcp import FastMCP

from backend.app.config import settings

mcp = FastMCP("tdas-retrieval")

_retriever = None


def _debug(msg: str):
    with open(_LOGDIR / "mcp_server.log", "a", encoding="utf-8", errors="replace") as f:
        f.write(f"[DEBUG {msg}]\n")


def _get_retriever():
    """懒初始化 retriever。预热在主线程完成（Windows 上非主线程 import 会死锁）。"""
    global _retriever
    if _retriever is None:
        _debug("importing Retriever in thread=%s" % threading.current_thread().name)
        from backend.app.rag.retriever import Retriever
        _retriever = Retriever(repo=settings.target_repo)
        _debug("Retriever created")
    return _retriever


def _hits_to_json(hits: list[dict]) -> str:
    out = []
    for h in hits:
        p = h.get("payload") or {}
        out.append({
            "number": p.get("number"),
            "type": p.get("source_type"),
            "status": p.get("status"),
            "author": p.get("author"),
            "url": p.get("url"),
            "score": round(float(h.get("score") or 0), 3),
            "snippet": (h.get("text") or "")[:500],
        })
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """在 LangChain 官方文档中检索技术主题（语义+关键词混合）。返回命中片段 JSON。"""
    return _hits_to_json(_get_retriever().search(query, limit=limit, source_type="doc"))


@mcp.tool()
def search_issues(query: str, status: str | None = None, limit: int = 5) -> str:
    """在 LangChain issue 讨论中检索技术问题，可按状态过滤（open/closed）。返回命中 JSON。"""
    filters = {"status": status} if status else None
    return _hits_to_json(
        _get_retriever().search(query, limit=limit, source_type="issue", filters=filters)
    )


@mcp.tool()
def search_prs(query: str, status: str | None = None, limit: int = 5) -> str:
    """在 LangChain PR 及其讨论中检索代码级证据（修复/重构/行为变更）。返回命中 JSON。"""
    filters = {"status": status} if status else None
    return _hits_to_json(
        _get_retriever().search(query, limit=limit, source_type="pr", filters=filters)
    )


@mcp.tool()
def get_issue_detail(number: int) -> str:
    """按编号获取某个 issue/PR 的完整正文与评论（纯元数据过滤）。返回 JSON。"""
    hits = _get_retriever().get_by_number(number)
    out = [{
        "number": h["payload"].get("number"),
        "type": h["payload"].get("source_type"),
        "kind": h["payload"].get("kind"),
        "author": h["payload"].get("author"),
        "text": (h.get("text") or "")[:2000],
    } for h in hits]
    return json.dumps(out, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TDAS 检索层 MCP server")
    ap.add_argument("--transport", choices=["stdio", "sse", "http"], default="stdio",
                    help="stdio=客户端拉起（启动慢~1min）；sse/http=常驻服务（Claude Code 等客户端推荐）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # 主线程预热：Windows 上任何非主线程首次 import torch/sklearn 都会与 asyncio 事件循环
    # 死锁（实测），必须在 mcp.run 之前主线程完成。启动 ~1min 是必要成本 —— 这就是为什么
    # Claude Code 等客户端要用 SSE 连常驻服务（客户端拉起的 30s 超时永远不够）。
    _debug("warm-up: start (main thread)")
    _get_retriever()
    _debug("warm-up: done")
    # show_banner=False：ASCII banner 含 UTF-8 表情，Windows 客户端（GBK stderr）读它会挂。
    mcp.run(transport=args.transport, host=args.host, port=args.port, show_banner=False)
