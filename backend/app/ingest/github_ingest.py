"""GitHub 数据拉取（GraphQL 主路径）。

- 主路径 `pull_repo_graphql`：游标分页，一次拉全量 issues+PRs 正文 + 评论，
  规避 REST `/issues` 接口「100 页 / 1 万条」的硬上限；幂等可续（状态文件）。
- 备用 `pull_issues_prs`（REST）：仅适合 <1 万条的小仓 spike。

数据源是「数据不是指令」：后续 agent 读取时按文本处理，禁止当指令执行。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import requests

from ..config import settings

_UA = {"Accept": "application/vnd.github+json"}


class GitHubClient:
    def __init__(self, token: str | None = None):
        self.token = token or settings.github_token
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"token {self.token}", **_UA})

    def _get(self, url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
        for i in range(retries):
            r = self.s.get(url, params=params, timeout=30)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers.get("Retry-After") or 60)
                print(f"[REST 限流] sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return r
            if r.status_code >= 500 and i < retries - 1:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r
        return r


# ---------------- GraphQL（主路径） ----------------

def _graphql(client: GitHubClient, query: str, variables: dict | None = None) -> dict:
    body = {"query": query}
    if variables:
        body["variables"] = variables
    r = client.s.post("https://api.github.com/graphql", json=body, timeout=120)
    if r.status_code in (403, 429):
        print("[GraphQL 403/429] sleep 60s", flush=True)
        time.sleep(60)
        return _graphql(client, query, variables)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        print("[GraphQL errors]", str(d["errors"][:3])[:800], flush=True)
    return d


def _until_reset(ext: dict) -> float:
    ra = ext.get("resetAt")
    if not ra:
        return 300.0
    try:
        reset = datetime.fromisoformat(ra.replace("Z", "+00:00"))
        return max(0.0, (reset - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 300.0


def _maybe_wait(data: dict):
    """按 data.rateLimit 自适应退避。实测大查询 cost≈1，通常不触发。"""
    rl = ((data.get("data") or {}).get("rateLimit")) or {}
    remaining, cost = rl.get("remaining", 5000), rl.get("cost", 0)
    if remaining < max(200, cost * 4):
        wait = _until_reset(rl)
        print(f"\n[GraphQL 限流] cost={cost} remaining={remaining} -> sleep {wait:.0f}s", flush=True)
        time.sleep(wait)


def _repo_batch_query(owner: str, repo: str, conn: str, batch: int, comments_per: int,
                      cursor: str | None) -> tuple[str, dict]:
    comments_field = (
        f"comments(first: {comments_per}) {{ nodes {{ id author {{ login }} createdAt body }} }}"
    )
    q = f"""
query($owner:String!,$repo:String!,$cursor:String){{
  rateLimit {{ cost remaining resetAt }}
  repository(owner:$owner, name:$repo){{
    {conn}(first: {batch}, after: $cursor, orderBy: {{field: CREATED_AT, direction: DESC}}){{
      pageInfo {{ hasNextPage endCursor }}
      nodes {{
        number title body state createdAt closedAt url
        author {{ login }}
        labels(first: 10) {{ nodes {{ name }} }}
        {comments_field}
      }}
    }}
  }}
}}"""
    return q, {"owner": owner, "repo": repo, "cursor": cursor}


def _graphql_to_record(nd: dict, kind: str) -> dict:
    return {
        "id": nd.get("id"),
        "number": nd.get("number"),
        "type": kind,  # "issue" | "pr"
        "title": nd.get("title") or "",
        "state": nd.get("state"),
        "created_at": nd.get("createdAt"),
        "closed_at": nd.get("closedAt"),
        "user": (nd.get("author") or {}).get("login"),
        "labels": [l["name"] for l in (nd.get("labels") or {}).get("nodes") or []],
        "body": nd.get("body") or "",
        "url": nd.get("url"),
    }


def _graphql_to_comment(c: dict, number: int, kind: str, owner: str, repo: str) -> dict:
    return {
        "comment_id": c.get("id"),
        "number": number,
        "source_type": kind,
        "author": (c.get("author") or {}).get("login"),
        "created_at": c.get("createdAt"),
        "body": c.get("body") or "",
        "url": f"https://github.com/{owner}/{repo}/issues/{number}",
    }


def _load_state(out_dir: str) -> dict:
    p = os.path.join(out_dir, "_pull_state.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(out_dir: str, state: dict):
    with open(os.path.join(out_dir, "_pull_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)


def pull_repo_graphql(owner: str, repo: str, token: str | None = None, out_dir: str | None = None,
                      batch: int = 100, comments_per: int = 50, max_items: int | None = None,
                      force: bool = False) -> tuple[int, int]:
    """GraphQL 全量拉取：issues+PRs 正文（issues_prs.jsonl）+ 评论（comments.jsonl）。

    无 REST 1 万条上限；幂等可续（_pull_state.json 记录游标）。返回 (正文数, 评论数)。
    """
    out_dir = out_dir or os.path.join(settings.raw_data_dir, f"{owner}__{repo}")
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "issues_prs.jsonl")
    comments_path = os.path.join(out_dir, "comments.jsonl")
    if force:
        for p in (items_path, comments_path, os.path.join(out_dir, "_pull_state.json")):
            if os.path.exists(p):
                os.remove(p)

    state = {} if force else _load_state(out_dir)
    client = GitHubClient(token)
    total_body = total_comment = 0
    for kind, conn in (("issue", "issues"), ("pr", "pullRequests")):
        if (state.get(kind) or {}).get("done"):
            print(f"{conn} 已完成，跳过", flush=True)
            continue
        cursor = (state.get(kind) or {}).get("cursor")
        done = False
        while True:
            q, v = _repo_batch_query(owner, repo, conn, batch, comments_per, cursor)
            data = _graphql(client, q, v)
            _maybe_wait(data)
            conn_data = ((data.get("data") or {}).get("repository") or {}).get(conn) or {}
            page_info = conn_data.get("pageInfo") or {}
            nodes = conn_data.get("nodes") or []
            with open(items_path, "a", encoding="utf-8") as fi, open(comments_path, "a", encoding="utf-8") as fc:
                for nd in nodes:
                    fi.write(json.dumps(_graphql_to_record(nd, kind), ensure_ascii=False) + "\n")
                    total_body += 1
                    for c in (nd.get("comments") or {}).get("nodes") or []:
                        fc.write(json.dumps(_graphql_to_comment(c, nd.get("number"), kind, owner, repo), ensure_ascii=False) + "\n")
                        total_comment += 1
            print(f"\r  {conn}: 正文={total_body} 评论={total_comment}", end="", flush=True)
            if not page_info.get("hasNextPage"):
                done = True
                break
            cursor = page_info.get("endCursor")
            state.setdefault(kind, {})["cursor"] = cursor
            _save_state(out_dir, state)
            if max_items and total_body >= max_items:
                break
        if done:
            state.setdefault(kind, {})["done"] = True
            _save_state(out_dir, state)
        print()
    return total_body, total_comment


# ---------------- REST（备用，仅 <1 万条小仓） ----------------

def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _to_record(it: dict) -> dict:
    return {
        "id": it.get("id"),
        "number": it.get("number"),
        "type": "pr" if it.get("pull_request") else "issue",
        "title": it.get("title") or "",
        "state": it.get("state"),
        "created_at": it.get("created_at"),
        "updated_at": it.get("updated_at"),
        "closed_at": it.get("closed_at"),
        "user": (it.get("user") or {}).get("login"),
        "labels": [l["name"] for l in it.get("labels", [])],
        "comments_count": it.get("comments", 0),
        "body": it.get("body") or "",
        "url": it.get("html_url"),
    }


def pull_issues_prs(owner: str, repo: str, token: str | None = None,
                    out_path: str | None = None, per_page: int = 100,
                    max_items: int | None = None, force: bool = False) -> int:
    """REST 拉 issues+PRs（受 1 万条上限，仅小仓/采样用）。"""
    out_path = out_path or os.path.join(settings.raw_data_dir, f"{owner}__{repo}", "issues_prs.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if force and os.path.exists(out_path):
        os.remove(out_path)
    n0 = 0 if force else _count_lines(out_path)
    client = GitHubClient(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    page, skipped, written = 1, 0, n0
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            r = client._get(url, params={"state": "all", "per_page": per_page, "page": page})
            if r.status_code != 200:
                break
            items = r.json()
            if not items:
                break
            for it in items:
                if skipped < n0:
                    skipped += 1
                    continue
                f.write(json.dumps(_to_record(it), ensure_ascii=False) + "\n")
                written += 1
            page += 1
            print(f"\r  [page {page - 1}] 累计 {written} 条", end="", flush=True)
            if max_items and written - n0 >= max_items:
                break
    print()
    return written


def pull_comments_graphql(owner: str, repo: str, token: str | None = None,
                          items_path: str | None = None, out_path: str | None = None,
                          batch_size: int = 100, comments_per: int = 100,
                          skip_existing: bool = True) -> int:
    """备用：对 REST 已拉列表补评论（主路径已并入全量拉取，一般不需要）。"""
    items_path = items_path or os.path.join(settings.raw_data_dir, f"{owner}__{repo}", "issues_prs.jsonl")
    out_path = out_path or os.path.join(os.path.dirname(items_path), "comments.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pairs: list[tuple[str, int]] = []
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pairs.append((rec.get("type") or "issue", rec.get("number")))

    client = GitHubClient(token)
    n = 0
    total = len(pairs)
    for start in range(0, total, batch_size):
        batch = pairs[start:start + batch_size]
        q = _comments_batch_query(owner, repo, batch, comments_per)
        data = _graphql(client, q)
        _maybe_wait(data)
        repo_data = (data.get("data") or {}).get("repository") or {}
        with open(out_path, "a", encoding="utf-8") as f:
            for key, val in repo_data.items():
                if not val:
                    continue
                is_issue = key.startswith("i")
                number = int(key[1:])
                for c in (val.get("comments") or {}).get("nodes") or []:
                    f.write(json.dumps(_graphql_to_comment(
                        c, number, "issue" if is_issue else "pr", owner, repo), ensure_ascii=False) + "\n")
                    n += 1
        print(f"\r  comments: {n}（batch {start // batch_size + 1} / {(total + batch_size - 1) // batch_size}）", end="", flush=True)
    print()
    return n


def _comments_batch_query(owner: str, repo: str, pairs: list[tuple[str, int]],
                          comments_per: int = 100) -> str:
    fields = f"comments(first: {comments_per}) {{ nodes {{ id author {{ login }} createdAt body }} }}"
    aliases = []
    for kind, number in pairs:
        prefix = "i" if kind == "issue" else "p"
        field = "issue" if kind == "issue" else "pullRequest"
        aliases.append(f"{prefix}{number}: {field}(number: {number}) {{ {fields} }}")
    return f'query {{ repository(owner: "{owner}", name: "{repo}") {{ {" ".join(aliases)} }} }}'
