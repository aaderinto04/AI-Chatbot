# Personal Knowledge Base (RAG) Chatbot

This project provides a small Retrieval-Augmented Generation (RAG) chatbot:
- Upload PDFs and text files
- Extract + smart chunk the content
- Create OpenAI embeddings and store them in a persistent local FAISS index
- Retrieve the most relevant chunks for each question
- Answer using ONLY the retrieved context (otherwise returns `I don't know`)

## Project Structure

- `app/main.py` - FastAPI server + API endpoints
- `app/ingest.py` - document parsing + smart chunking
- `app/embed.py` - OpenAI embeddings
- `app/retrieve.py` - retrieval + context-only prompting
- `app/utils.py` - persistent FAISS store + helpers
- `data/` - uploaded files
- `db/` - FAISS index + chunk metadata

## Local Setup

1. Install Python dependencies

```bash
cd /Users/abdullahaderinto/Documents/AI-Chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables

```bash
export OPENAI_API_KEY="your_key_here"
```

Optional tuning (defaults shown):

```bash
export CHUNK_SIZE_TOKENS=900
export CHUNK_OVERLAP_TOKENS=300
export OPENAI_CHAT_MODEL=gpt-4o-mini
```

3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Server endpoints:
- `POST /upload`
- `POST /query`
- `GET /health`

## API Usage

### Upload documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@/path/to/file.pdf" \
  -F "files=@/path/to/notes.txt"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the document say about X?",
    "session_id": "user-1",
    "top_k": 5,
    "document_names": null,
    "use_chat_history": true
  }'
```

Response format:

```json
{
  "answer": "....",
  "sources": [
    { "file": "my_document.pdf", "snippet": "..." }
  ]
}
```

### Filter by specific documents (bonus)

Send `document_names` as the stored/sanitized filenames (the backend uses a safe filename transformation).
Example:

```json
{
  "question": "Only from section 2",
  "document_names": ["section_2.txt"]
}
```

## Notes / Limitations

- This implementation stores chunk text + metadata in `db/chunks.json` for simplicity.
- Very large document uploads may take time due to embedding calls.