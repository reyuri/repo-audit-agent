"""集中配置：读 .env（pydantic-settings）。"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM（OpenAI 兼容，可切换 provider）----
    # 具体 key/base_url/模型名都在私有 .env 里配置；代码里只按角色取模型。
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model_fast: str = ""       # 检索 Worker / 一般开发（低成本快速模型）
    llm_model_reasoning: str = ""  # 编排/反思（强推理模型）

    # ---- GitHub ----
    github_token: str = ""

    # ---- 存储 ----
    qdrant_path: str = str(ROOT / "data" / "qdrant")
    db_path: str = str(ROOT / "data" / "raat.db")
    raw_data_dir: str = str(ROOT / "data" / "raw")
    cache_dir: str = str(ROOT / "data" / "cache")

    # ---- 索引 ----
    target_repo: str = "langchain-ai/langchain"
    # 官方文档独立仓库（LangChain 文档已拆出主仓）与内容前缀
    docs_repo: str = "langchain-ai/docs"
    docs_prefix: str = "src/"
    # 默认 all-MiniLM-L6-v2（384 维，torch CPU 快）；可换 BAAI/bge-base-en-v1.5（768 维）
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_batch_size: int = 64

    # 检索
    top_k: int = 10
    rrf_k: int = 60  # RRF 融合常数

    # ---- 网络 ----
    # HuggingFace 国内镜像（huggingface.co 连不上）；留空则走官方
    hf_endpoint: str = "https://hf-mirror.com"

    # ---- 可观测（LangSmith，key-gated）----
    # 有 key 才激活追踪（LangGraph 自动图级 + LLM 方法 @traceable span 级）；
    # 无 key 时静默降级、不影响运行。
    langsmith_api_key: str = ""
    langchain_project: str = "raat"


settings = Settings()

# 尽早写入环境变量，保证 fastembed/huggingface_hub 走镜像下载
if settings.hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

# LangSmith 追踪：只有配置了 API key 才开启，避免空 key 时每次调用都打 API
# 拖慢运行。LangGraph/langsmith 读标准环境变量激活。
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)


def data_path(key: str) -> Path:
    """把配置里（可能是相对的）路径解析到项目根下。"""
    raw = getattr(settings, key)
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p
