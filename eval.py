"""
Reference-free RAGAS evaluation for the cheer rules RAG pipeline.

Metrics (no ground truth needed):
  - Faithfulness:                    does the answer only say things supported by the retrieved chunks?
  - LLMContextPrecisionWithoutReference: were the retrieved chunks actually relevant to the question?

Run with:
    python eval.py
    python eval.py --output results.json   # save scores to file
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

from src.retriever import ask, retrieve

load_dotenv()

FEEDBACK_LOG = Path("feedback.jsonl")

SEED_QUESTIONS = [
    # Legality — single skill
    "Is a back handspring legal at level 2?",
    "Can I do a full twisting layout at level 5?",
    "Is an extended liberty legal at level 3?",
    "Can a level 1 team do a basket toss?",
    # Legality — sequence
    "Can I do a round off back handspring back tuck at level 3?",
    "Is it legal to load to extension without a spotter at level 4?",
    # Skill listing
    "What tumbling skills are allowed at level 2?",
    "What stunts can I do at level 1?",
    # Definition
    "What is a bracer?",
    "What is an inversion?",
]


def _load_feedback_questions() -> list[str]:
    if not FEEDBACK_LOG.exists():
        return []
    questions = []
    with FEEDBACK_LOG.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                q = entry.get("question", "").strip()
                if q:
                    questions.append(q)
            except json.JSONDecodeError:
                continue
    return questions


def _build_sample(question: str) -> dict | None:
    try:
        result = ask(question)
    except Exception as e:
        print(f"  [skip] '{question[:60]}' — ask() failed: {e}")
        return None

    try:
        chunks = retrieve(
            result.get("search_query", question),
            category=result.get("category"),
            level_group=result.get("level_group"),
        )
        contexts = [c["text"] for c in chunks]
    except Exception as e:
        print(f"  [skip] '{question[:60]}' — retrieve() failed: {e}")
        return None

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": contexts,
    }


def build_dataset(questions: list[str]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": []}
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q[:70]}")
        sample = _build_sample(q)
        if sample is not None:
            rows["question"].append(sample["question"])
            rows["answer"].append(sample["answer"])
            rows["contexts"].append(sample["contexts"])
    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser(description="Run reference-free RAGAS eval.")
    parser.add_argument("--output", type=Path, default=None, help="Save scores to this JSON file.")
    parser.add_argument("--feedback-only", action="store_true", help="Only use questions from feedback.jsonl.")
    args = parser.parse_args()

    feedback_qs = _load_feedback_questions()
    if args.feedback_only:
        questions = feedback_qs
    else:
        seen = set()
        questions = []
        for q in feedback_qs + SEED_QUESTIONS:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                questions.append(q)

    if not questions:
        print("No questions found. Add questions to SEED_QUESTIONS or run the app to collect feedback.")
        return

    print(f"\nBuilding dataset from {len(questions)} questions...")
    dataset = build_dataset(questions)

    if len(dataset) == 0:
        print("All questions failed — nothing to evaluate.")
        return

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)

    print(f"\nRunning RAGAS eval on {len(dataset)} samples...")
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()],
        llm=llm,
    )

    print("\n--- Results ---")
    df = result.to_pandas()
    skip_cols = {"question", "answer", "contexts", "user_input", "response", "retrieved_contexts"}
    numeric_cols = [c for c in df.columns if c not in skip_cols]
    if not numeric_cols:
        print("No metric columns found. Raw columns:", list(df.columns))
        return
    for metric in numeric_cols:
        score = df[metric].mean()
        print(f"  {metric:<45} {score:.3f}")

    if args.output:
        args.output.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
