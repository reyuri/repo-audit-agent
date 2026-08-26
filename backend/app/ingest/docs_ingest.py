"""官方文档抓取：GitHub git tree 列文件 + raw 拉内容（CDN 不计 API 配额）。

- 列表：GET /repos/{o}/{r}/git/trees/{branch}?recursive=1（1 次请求）
- 内容：raw.githubusercontent.com（CDN，不占用 API rate limit）
- 分块：见 indexer.chunk_doc_text（按 markdown 标题切）
"""
from __future__ import annotations

import os
import time

import requests

from ..config import settings
from .github_ingest import GitHubClient


def get_default_branch(client: GitHubClient, owner: str, repo: str) -> str:
    r = client.s.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=30)
    r.raise_for_status()
    return r.json().get("default_branch", "main")


def list_doc_files(owner: str, repo: str, token: str | None = None,
                   prefix: str | None = None, exts: tuple = (".md", ".mdx")) -> list[str]:
    prefix = prefix or settings.docs_prefix
    """返回 docs/ 下所有 markdown 文件路径。"""
    client = GitHubClient(token)
    branch = get_default_branch(client, owner, repo)
    r = client._get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
                    params={"recursive": "1"})
    r.raise_for_status()
    data = r.json()
    if data.get("truncated"):
        print("[warn] git tree truncated，可能漏文件", flush=True)
    files = []
    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path.startswith(prefix):
            continue
        if not path.lower().endswith(exts):
            continue
        files.append(path)
    return files


def fetch_raw(owner: str, repo: str, branch: str, path: str, session: requests.Session,
              retries: int = 4) -> str | None:
    """raw.githubusercontent.com 拉原文（国内网络不稳，加重试退避）；彻底失败返回 None。"""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == retries - 1:
                print(f"[raw fetch 失败] {path}: {type(e).__name__}", flush=True)
                return None
            time.sleep(1 + attempt * 2)  # 1s, 3s, 5s 退避
    return None


def fetch_all_docs(owner: str, repo: str, token: str | None = None,
                   limit: int | None = None, concurrency: int = 8) -> tuple[list[dict], str]:
    """拉取全部文档原文。返回 [{path, text}]。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = GitHubClient(token)
    branch = get_default_branch(client, owner, repo)
    paths = list_doc_files(owner, repo, token)
    if limit:
        paths = paths[:limit]
    print(f"共 {len(paths)} 个文档文件（branch={branch}）", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "tdas-doc-ingest"})
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(fetch_raw, owner, repo, branch, p, session): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            text = fut.result()
            if text:
                records.append({"path": p, "text": text})
            done += 1
            if done % 50 == 0:
                print(f"\r  已抓取 {done}/{len(paths)}", end="", flush=True)
    print()
    return records, branch


def save_docs_jsonl(records: list[dict], owner: str, repo: str) -> str:
    """把 chunk 后的文档存 docs.jsonl（供索引 + BM25 复用）。"""
    import json

    from ..rag.indexer import chunk_doc_text

    out_dir = os.path.join(settings.raw_data_dir, f"{owner}__{repo}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "docs.jsonl")
    full_repo = f"{owner}/{repo}"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            for d in chunk_doc_text(rec["text"], rec["path"], repo=full_repo):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
    return out_path, n
