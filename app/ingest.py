import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tiktoken
from pypdf import PdfReader

from .embed import embed_texts, get_embedding_model_name
from .utils import FaissStore, safe_filename


DEFAULT_CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "900"))
DEFAULT_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "300"))


def _get_encoder() -> tiktoken.Encoding:
    # Try to get model-specific encoding for more consistent token counts.
    emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        return tiktoken.encoding_for_model(emb_model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(encoder: tiktoken.Encoding, text: str) -> int:
    return len(encoder.encode(text))


def _split_paragraphs(text: str) -> List[str]:
    # Normalize newlines and split by blank lines.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in normalized.split("\n\n")]
    return [p for p in parts if p]


def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Returns list of (page_num, page_text).
    """

    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def extract_text_from_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _split_long_unit_to_token_chunks(
    unit_text: str,
    encoder: tiktoken.Encoding,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    tokens = encoder.encode(unit_text)
    if len(tokens) <= chunk_size_tokens:
        return [unit_text]

    out: List[str] = []
    start = 0
    step = max(chunk_size_tokens - overlap_tokens, 1)
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        seg_tokens = tokens[start:end]
        out.append(encoder.decode(seg_tokens))
        if end >= len(tokens):
            break
        start += step
    return out


def chunk_paragraphs_smart(
    *,
    paragraphs: Sequence[Dict[str, Any]],
    filename: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Create chunks by grouping paragraphs until chunk_size_tokens is reached.
    Overlap is applied by reusing paragraph units whose token totals cover overlap_tokens.
    """

    encoder = _get_encoder()

    # Expand very large paragraphs into token-based segments (treated as paragraphs).
    units: List[Dict[str, Any]] = []
    for p in paragraphs:
        text = p["text"]
        page = p.get("page")
        token_count = _count_tokens(encoder, text)
        if token_count > chunk_size_tokens:
            segments = _split_long_unit_to_token_chunks(
                text,
                encoder=encoder,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
            )
            for seg in segments:
                units.append({"text": seg, "page": page})
        else:
            units.append({"text": text, "page": page})

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    def unit_tokens(u: Dict[str, Any]) -> int:
        return _count_tokens(encoder, u["text"])

    i = 0
    while i < len(units):
        cur_units: List[Dict[str, Any]] = []
        cur_tokens = 0
        start_i = i

        while i < len(units):
            u = units[i]
            u_tok = unit_tokens(u)
            if cur_units and (cur_tokens + u_tok) > chunk_size_tokens:
                break
            cur_units.append(u)
            cur_tokens += u_tok
            i += 1

        # Create a chunk from cur_units.
        text = "\n\n".join([u["text"] for u in cur_units]).strip()
        if text:
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "chunk_index": chunk_idx,
                        "page_start": cur_units[0].get("page"),
                    },
                }
            )
            chunk_idx += 1

        if i >= len(units):
            break

        # Move i back to create overlap. We'll find a new start that covers overlap_tokens.
        # (We already consumed up to i, so overlap is from the end of cur_units.)
        overlap_target = overlap_tokens
        back_tokens = 0
        j = len(cur_units) - 1
        while j >= 0 and back_tokens < overlap_target:
            back_tokens += unit_tokens(cur_units[j])
            j -= 1

        # Next chunk should start at: start of cur_units + (j+1)
        next_start = start_i + (j + 1)
        # Ensure forward progress.
        if next_start <= start_i:
            next_start = start_i + 1
        i = next_start

    return chunks


def build_paragraph_units_from_pdf(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for page_num, page_text in pages:
        for para in _split_paragraphs(page_text):
            units.append({"text": para, "page": page_num})
    return units


def build_paragraph_units_from_text(text: str) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for para in _split_paragraphs(text):
        units.append({"text": para, "page": None})
    return units


def process_and_ingest_files(
    *,
    file_paths: List[Path],
    store: FaissStore,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Dict[str, Any]:
    """
    Parse + chunk + embed + add to FAISS store.
    Skips files that already exist in the store by matching `metadata.source`.
    """

    added_sources: List[str] = []
    skipped_sources: List[str] = []
    total_chunks_added = 0

    for file_path in file_paths:
        source = safe_filename(file_path.name)
        if store.has_source(source):
            skipped_sources.append(source)
            continue

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            pages = extract_text_from_pdf(file_path)
            paragraphs = build_paragraph_units_from_pdf(pages)
        elif ext in {".txt", ".md", ".text"}:
            text = extract_text_from_text_file(file_path)
            paragraphs = build_paragraph_units_from_text(text)
        else:
            # Fallback: try text extraction.
            text = extract_text_from_text_file(file_path)
            paragraphs = build_paragraph_units_from_text(text)

        chunks = chunk_paragraphs_smart(
            paragraphs=paragraphs,
            filename=source,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

        if not chunks:
            continue

        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        store.embedding_model = get_embedding_model_name()
        store.add(embeddings, chunks)
        store.save()

        added_sources.append(source)
        total_chunks_added += len(chunks)

    return {
        "added_sources": added_sources,
        "skipped_sources": skipped_sources,
        "total_chunks_added": total_chunks_added,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": overlap_tokens,
    }

