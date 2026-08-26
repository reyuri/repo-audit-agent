"""全量拉取 LangChain issues+PRs（正文+评论，GraphQL 游标分页，无 1 万条上限）。

用法（项目根目录，conda activate tdas）：
    python -m backend.scripts.ingest_repo                 # 全量（断点续传）
    python -m backend.scripts.ingest_repo --limit 2000    # 抽样（spike 用）
    python -m backend.scripts.ingest_repo --force         # 清空重拉
"""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK 控制台兼容

from ..app.config import data_path, settings
from ..app.ingest.github_ingest import pull_repo_graphql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="最多拉 N 条正文（spike 用）")
    ap.add_argument("--force", action="store_true", help="清空已拉数据重来")
    args = ap.parse_args()

    owner, repo = settings.target_repo.split("/")
    out_dir = str(data_path("raw_data_dir") / f"{owner}__{repo}")
    body, comments = pull_repo_graphql(owner, repo, out_dir=out_dir,
                                       max_items=args.limit, force=args.force)
    print(f"✅ 完成：正文 {body} 条，评论 {comments} 条 -> {out_dir}")


if __name__ == "__main__":
    main()
