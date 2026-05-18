"""
FastAPI wrapper around the cheer rules RAG retriever.

Run with:
    uvicorn api:app --reload
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.retriever import ask

app = FastAPI(title="Cheer Rules AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

FEEDBACK_LOG = Path("feedback.jsonl")


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
    return ask(body.query)


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
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}
