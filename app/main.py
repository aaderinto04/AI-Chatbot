import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ingest import process_and_ingest_files
from .retrieve import answer_with_context
from .utils import DATA_DIR, FaissStore, safe_filename


app = FastAPI(title="Personal Knowledge Base (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-memory multi-turn chat history (bonus).
CHAT_HISTORIES: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))


class UploadResponse(BaseModel):
    added_sources: List[str]
    skipped_sources: List[str]
    total_chunks_added: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    document_names: Optional[List[str]] = Field(
        default=None,
        description="Optional list of filenames to restrict retrieval. Should match stored names (sanitized).",
    )
    use_chat_history: bool = Field(default=True)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]] = Field(default_factory=list)


store: Optional[FaissStore] = None


@app.on_event("startup")
def _load_store() -> None:
    global store
    store = FaissStore.load()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "index_loaded": store is not None and not store.is_empty}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    assert store is not None, "Store not initialized."
    if not files:
        raise ValueError("No files uploaded.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_paths: List[Path] = []
    for f in files:
        if not f.filename:
            continue
        sanitized = safe_filename(f.filename)
        out_path = DATA_DIR / sanitized
        # Always write the file so ingestion can read it; ingest will skip if already indexed.
        raw = await f.read()
        out_path.write_bytes(raw)
        file_paths.append(out_path)

    result = process_and_ingest_files(
        file_paths=file_paths,
        store=store,
    )
    return UploadResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    assert store is not None, "Store not initialized."

    session_id = req.session_id or "default"
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = []

    filter_sources: Optional[List[str]] = None
    if req.document_names:
        # Sanitization should match how ingest stores `metadata.source`.
        filter_sources = [safe_filename(n) for n in req.document_names]

    history = CHAT_HISTORIES.get(session_id, [])
    chat_history_for_model: Optional[List[Dict[str, str]]] = None
    if req.use_chat_history and history:
        chat_history_for_model = history

    result = answer_with_context(
        query=req.question,
        store=store,
        top_k=req.top_k,
        filter_sources=filter_sources,
        chat_history=chat_history_for_model,
    )

    answer = result["answer"]
    sources = result.get("sources") or []

    # Update in-memory history.
    if req.use_chat_history:
        CHAT_HISTORIES[session_id].append({"role": "user", "content": req.question})
        CHAT_HISTORIES[session_id].append({"role": "assistant", "content": answer})
        CHAT_HISTORIES[session_id] = CHAT_HISTORIES[session_id][-MAX_HISTORY_MESSAGES:]

    return QueryResponse(answer=answer, sources=sources)

