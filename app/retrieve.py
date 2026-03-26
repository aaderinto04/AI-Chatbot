import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from openai import OpenAI

from .embed import embed_texts
from .utils import FaissStore, RetrievedChunk


CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def retrieve_chunks(
    *,
    query: str,
    store: FaissStore,
    top_k: int = 5,
    filter_sources: Optional[Sequence[str]] = None,
    candidate_multiplier: int = 10,
) -> List[RetrievedChunk]:
    """
    Returns top-k chunks by cosine similarity.

    Bonus: if filter_sources is provided, we retrieve a larger candidate set
    then filter in-memory by metadata.source.
    """

    if store.is_empty:
        return []

    query_embeddings = embed_texts([query])
    if query_embeddings.shape[0] != 1:
        return []

    # Candidate retrieval first (helps filtering).
    candidate_k = max(top_k * candidate_multiplier, top_k)
    scores_and_ids = store.search(query_embedding=query_embeddings, top_k=candidate_k)

    filter_set: Optional[Set[str]] = None
    if filter_sources:
        filter_set = {s for s in filter_sources if s}

    results: List[RetrievedChunk] = []
    for chunk_global_idx, score in scores_and_ids:
        chunk = store.get_chunk(chunk_global_idx)
        metadata = chunk.get("metadata", {})
        if filter_set is not None:
            if metadata.get("source") not in filter_set:
                continue
        results.append(
            RetrievedChunk(
                text=chunk.get("text", ""),
                metadata=metadata,
                score=score,
            )
        )
        if len(results) >= top_k:
            break

    return results


def _format_context(context_chunks: Sequence[RetrievedChunk]) -> str:
    blocks: List[str] = []
    for i, c in enumerate(context_chunks, start=1):
        source = c.metadata.get("source", "unknown")
        chunk_index = c.metadata.get("chunk_index", "unknown")
        page_start = c.metadata.get("page_start")
        page_part = f", page {page_start}" if page_start else ""
        blocks.append(
            f"[{i}] (source: {source}, chunk: {chunk_index}{page_part})\n{c.text}"
        )
    return "\n\n".join(blocks).strip()


def _postprocess_answer(text: str) -> str:
    cleaned = (text or "").strip()
    norm = cleaned.replace("’", "'").strip()
    # Accept minor variations from the model.
    norm = norm.rstrip(".")
    if norm.lower() == "i don't know":
        return "I don't know"
    return cleaned


def answer_with_context(
    *,
    query: str,
    store: FaissStore,
    top_k: int = 5,
    filter_sources: Optional[Sequence[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Retrieves context and asks the LLM to answer ONLY from that context.
    """

    retrieved = retrieve_chunks(
        query=query,
        store=store,
        top_k=top_k,
        filter_sources=filter_sources,
    )

    if not retrieved:
        return {"answer": "I don't know", "sources": []}

    context = _format_context(retrieved)

    history_text = ""
    if chat_history:
        # Include only the last few turns to keep prompts small.
        last = chat_history[-6:]
        lines: List[str] = []
        for msg in last:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role.upper()}: {content}")
        history_text = "\n".join(lines).strip()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = (
        "You are a retrieval-based question answering assistant.\n"
        "Answer the user's question using ONLY the provided CONTEXT.\n"
        "Do not use prior knowledge, and do not guess.\n"
        "If the answer is not present in CONTEXT, reply with exactly: I don't know"
    )

    user_prompt_parts = [
        "CONTEXT:",
        context,
        "",
        "QUESTION:",
        query,
    ]
    if history_text:
        user_prompt_parts.insert(2, "CHAT HISTORY (for interpreting the question, not for facts):\n" + history_text)

    user_prompt = "\n".join(user_prompt_parts).strip()

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = _postprocess_answer(resp.choices[0].message.content or "")

    # Sources: return filenames + snippets from retrieved contexts.
    snippet_chars = int(os.getenv("SOURCE_SNIPPET_CHARS", "240"))
    sources_out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for c in retrieved:
        src = str(c.metadata.get("source", "unknown"))
        if src in seen:
            continue
        snippet = (c.text or "").strip()
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + "..."
        sources_out.append({"file": src, "snippet": snippet})
        seen.add(src)

    # If model says I don't know, return sources empty (optional but aligns to "not found").
    if answer == "I don't know":
        return {"answer": "I don't know", "sources": []}

    return {"answer": answer, "sources": sources_out}

