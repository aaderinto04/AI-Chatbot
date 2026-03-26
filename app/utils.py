import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"

FAISS_INDEX_PATH = DB_DIR / "index.faiss"
CHUNKS_PATH = DB_DIR / "chunks.json"
STORE_META_PATH = DB_DIR / "store_meta.json"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """
    Make filenames safe for writing under /data.
    Keeps extension; replaces path separators and weird characters.
    """

    name = os.path.basename(name)
    name = name.replace("\x00", "")
    name = re.sub(r"[\\/]+", "_", name)
    # Allow common filename characters, replace the rest.
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name or "upload"


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so that inner product == cosine similarity.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {x.shape}")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype("float32", copy=False)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float


class FaissStore:
    """
    Persistent store: FAISS index + chunk text/metadata.
    """

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        embedding_model: Optional[str] = None,
        dim: Optional[int] = None,
    ) -> None:
        self.index = index
        self.chunks = chunks or []
        self.embedding_model = embedding_model
        self.dim = dim

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def has_source(self, source: str) -> bool:
        return any(c.get("metadata", {}).get("source") == source for c in self.chunks)

    def ensure_index(self, dim: int) -> None:
        if self.index is None:
            # IndexFlatIP + L2-normalized vectors => cosine similarity.
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim
        elif self.dim is not None and self.dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: store dim={self.dim}, new dim={dim}"
            )

    def add(self, embeddings: np.ndarray, new_chunks: List[Dict[str, Any]]) -> None:
        if len(new_chunks) == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix (n, dim).")
        if embeddings.shape[0] != len(new_chunks):
            raise ValueError(
                f"Embeddings row count {embeddings.shape[0]} != chunk count {len(new_chunks)}"
            )

        self.ensure_index(dim=embeddings.shape[1])
        assert self.index is not None

        embeddings = normalize_embeddings(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    def save(self) -> None:
        ensure_dirs()

        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        else:
            # If empty, remove prior index if any.
            if FAISS_INDEX_PATH.exists():
                FAISS_INDEX_PATH.unlink()

        _write_json(CHUNKS_PATH, self.chunks)
        _write_json(
            STORE_META_PATH,
            {
                "embedding_model": self.embedding_model,
                "dim": self.dim,
                "ntotal": 0 if self.index is None else int(self.index.ntotal),
            },
        )

    @classmethod
    def load(cls, db_dir: Path = DB_DIR) -> "FaissStore":
        index_path = db_dir / "index.faiss"
        chunks_path = db_dir / "chunks.json"
        meta_path = db_dir / "store_meta.json"

        if not index_path.exists() and not chunks_path.exists():
            return cls(index=None, chunks=[], embedding_model=None, dim=None)

        chunks = _read_json(chunks_path, default=[])
        meta = _read_json(meta_path, default={})

        dim = meta.get("dim")
        embedding_model = meta.get("embedding_model")

        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            index = None

        return cls(index=index, chunks=chunks, embedding_model=embedding_model, dim=dim)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (chunk_global_index, score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must have shape (1, dim)")
        embeddings = normalize_embeddings(query_embedding)

        scores, idxs = self.index.search(embeddings, top_k)
        results: List[Tuple[int, float]] = []
        for chunk_idx, score in zip(idxs[0].tolist(), scores[0].tolist()):
            if chunk_idx == -1:
                continue
            results.append((int(chunk_idx), float(score)))
        return results

    def get_chunk(self, chunk_global_index: int) -> Dict[str, Any]:
        return self.chunks[chunk_global_index]

