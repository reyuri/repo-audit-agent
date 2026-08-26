"""SQLite：audit_memory（跨会话长期记忆）+ agent_traces（可观测轨迹）。

- audit_memory：审计过的仓库 / 上次审计结论 / 用户偏好 —— 跨会话增量复用。
- agent_traces：每个 worker 的 plan/reason/act/observe/reflect 都落这里，便于复盘「慢在哪、谁调了什么」。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .config import data_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_memory (
  repo TEXT PRIMARY KEY,
  last_audited_at TEXT,
  conclusions_json TEXT,
  user_prefs_json TEXT
);
CREATE TABLE IF NOT EXISTS agent_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT,
  agent TEXT,
  node TEXT,
  tool TEXT,
  latency_ms INTEGER,
  tokens INTEGER,
  detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_agent ON agent_traces(agent, ts);
CREATE TABLE IF NOT EXISTS doc_texts (
  point_id TEXT PRIMARY KEY,
  text TEXT NOT NULL
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DB:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_path("db_path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：D4 起 worker 用 ThreadPoolExecutor 并行跑，
        # 共享同一个 DB 连接跨线程写 trace；配合 self._lock 串行化写操作。
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def _write(self, sql: str, params: tuple):
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    # ---------- 长期记忆 ----------
    def upsert_audit_memory(self, repo: str, conclusions: dict, user_prefs: dict | None = None):
        self._write(
            "INSERT INTO audit_memory(repo, last_audited_at, conclusions_json, user_prefs_json) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(repo) DO UPDATE SET "
            "last_audited_at=excluded.last_audited_at, "
            "conclusions_json=excluded.conclusions_json, "
            "user_prefs_json=excluded.user_prefs_json",
            (repo, _now(), json.dumps(conclusions, ensure_ascii=False),
             json.dumps(user_prefs or {}, ensure_ascii=False)),
        )

    def get_audit_memory(self, repo: str) -> dict | None:
        row = self.conn.execute(
            "SELECT repo, last_audited_at, conclusions_json, user_prefs_json "
            "FROM audit_memory WHERE repo=?", (repo,)
        ).fetchone()
        if not row:
            return None
        return {
            "repo": row[0],
            "last_audited_at": row[1],
            "conclusions": json.loads(row[2] or "{}"),
            "user_prefs": json.loads(row[3] or "{}"),
        }

    # ---------- 可观测轨迹 ----------
    def trace(self, agent: str, node: str, tool: str = "", latency_ms: int = 0,
              tokens: int = 0, detail: str = ""):
        self._write(
            "INSERT INTO agent_traces(ts, agent, node, tool, latency_ms, tokens, detail) "
            "VALUES(?,?,?,?,?,?,?)",
            (_now(), agent, node, tool, int(latency_ms), int(tokens), detail[:2000]),
        )

    def recent_traces(self, limit: int = 50) -> list[tuple]:
        return self.conn.execute(
            "SELECT ts, agent, node, tool, latency_ms, tokens FROM agent_traces "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # ---------- 文档全文（与 Qdrant payload 分离，加快写入） ----------
    def put_doc_texts(self, mapping: dict[str, str]):
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO doc_texts(point_id, text) VALUES(?,?)",
                [(k, v) for k, v in mapping.items()],
            )
            self.conn.commit()

    def get_doc_texts(self, point_ids: list[str]) -> dict[str, str]:
        if not point_ids:
            return {}
        q = ",".join("?" * len(point_ids))
        rows = self.conn.execute(
            f"SELECT point_id, text FROM doc_texts WHERE point_id IN ({q})", point_ids
        ).fetchall()
        return dict(rows)
