import os
from typing import List

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _client() -> OpenAI:
    # OpenAI looks for OPENAI_API_KEY automatically, but being explicit is fine.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key)


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts via OpenAI embeddings API.
    Returns: numpy array of shape (n, dim) dtype float32.
    """

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    # Small batches reduce request payload size.
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "96"))

    client = _client()
    all_embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # OpenAI returns embeddings in the same order as inputs.
        all_embeddings.extend([d.embedding for d in resp.data])

    arr = np.array(all_embeddings, dtype="float32")
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding array shape: {arr.shape}")
    return arr


def get_embedding_model_name() -> str:
    return EMBEDDING_MODEL

