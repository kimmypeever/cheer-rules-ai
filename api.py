"""
FastAPI wrapper around the cheer rules RAG retriever.

Run with:
    uvicorn api:app --reload
"""

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


class QuestionRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_question(body: QuestionRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return ask(body.query)
