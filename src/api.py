import pickle
from typing import Dict, List, Optional

import faiss
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.utils import (
    extract_city,
    parse_date_range,
    build_rag_prompt,
    call_claude_stream,
    retrieve,
    fetch_events
)

app = FastAPI()

# Load FAISS index and chunks
try:
    index = faiss.read_index("faiss_index/index.faiss")
    chunks = pickle.load(open("faiss_index/chunks.pkl", "rb"))
except Exception as e:
    print(f"⚠️ Could not load FAISS index: {e}")
    index, chunks = None, []

embedder = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []


@app.get("/")
def root():
    return {"message": "✅ Events Chatbot API is running."}


@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query
    city = extract_city(query)
    start, end = parse_date_range(query)

    # Fetch events from external APIs
    event_docs = fetch_events(city, start, end)
    
    # If no events, fallback to Claude
    if not event_docs:
        prompt = f"You are a helpful events assistant. Answer this:\n\n{query}"
        stream = call_claude_stream(prompt)
        return StreamingResponse(content=stream, media_type="text/plain")
    
    # RAG: run vector search if index exists
    print(f"event docs is \n{event_docs}\n\n")
    docs = retrieve(query, event_docs, embedder, index) if index else event_docs[:5]
    print(f"docs is \n{docs}\n\n")
    context = "\n---\n".join(docs if isinstance(docs[0], str) else [
            f"{d['title']} — {d['start']} @ {d['location']}\n{d['url']}"
            for d in docs
        ]
    )
    print(f"context is \n{context}\n\n")

    rag_prompt = build_rag_prompt(context, query)
    print(f"rag prompt is \n{rag_prompt}")
    stream = call_claude_stream(rag_prompt)

    return StreamingResponse(content=stream, media_type="text/plain")