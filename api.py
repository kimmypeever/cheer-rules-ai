"""
FastAPI wrapper around the cheer rules RAG retriever.

Run with:
    uvicorn api:app --reload
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.retriever import ask

app = FastAPI(title="Cheer Rules AI")

_origins = ["http://localhost:3000"]
if _prod_origin := os.getenv("ALLOWED_ORIGIN"):
    _origins.append(_prod_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)

FEEDBACK_LOG = Path("feedback.jsonl")
FEEDBACK_BUCKET = os.getenv("FEEDBACK_BUCKET")
_FEEDBACK_BLOB = "feedback.jsonl"


def _write_feedback(entry: dict) -> None:
    line = json.dumps(entry) + "\n"
    if FEEDBACK_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        blob = client.bucket(FEEDBACK_BUCKET).blob(_FEEDBACK_BLOB)
        try:
            existing = blob.download_as_text(encoding="utf-8")
        except Exception:
            existing = ""
        blob.upload_from_string(existing + line, content_type="text/plain")
    else:
        with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


class QuestionRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "good" or "bad"
    note: str = ""


@app.post("/ask")
def ask_question(body: QuestionRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return ask(body.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    if body.rating not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="Rating must be 'good' or 'bad'.")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": body.rating,
        "question": body.question,
        "answer": body.answer,
        "note": body.note,
    }
    _write_feedback(entry)
    return {"status": "ok"}
