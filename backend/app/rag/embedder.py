"""Embedding：sentence-transformers（torch CPU，多线程 BLAS）。

> 为什么不用 fastembed/onnxruntime：本机实测 onnxruntime 病态慢且会挂死
> （bge-base 仅 ~11 docs/s，开图优化直接死锁）。torch CPU 多线程快 10-50×，
> 且规避了坏掉的 onnxruntime 环境。

默认模型 sentence-transformers/all-MiniLM-L6-v2（384 维，快）；
可切 BAAI/bge-base-en-v1.5（768 维，质量更好，速度慢些）。
"""
from __future__ import annotations

from ..config import settings


class Embedder:
    _inst: "Embedder | None" = None

    def __init__(self, model: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = model or settings.embed_model
        try:
            # 优先本地加载：st 6.0 的 PEFT adapter 检查会直连 huggingface.co（不走镜像），
            # 国内会超时重试导致假死。模型已缓存时 local_files_only 完全绕开网络。
            self.emb = SentenceTransformer(self.model, local_files_only=True)
        except Exception:
            self.emb = SentenceTransformer(self.model)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.emb.encode(
            texts,
            normalize_embeddings=True,
            batch_size=settings.embed_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
