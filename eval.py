"""
RAGAS evaluation for the cheer rules RAG pipeline.

Reference-free metrics (run on all questions):
  - Faithfulness:                        does the answer only cite things in the retrieved chunks?
  - LLMContextPrecisionWithoutReference: were the retrieved chunks relevant to the question?

Reference metrics (run only on questions where ground_truth is filled in):
  - ContextRecall:     do the retrieved chunks contain the information needed to answer?
  - AnswerCorrectness: does the answer match the ground truth?

Setup:
  Fill in "ground_truth" values in eval_questions.json using the actual rulebook.
  Leave ground_truth as "" to skip reference metrics for that question.

Usage:
    python eval.py
    python eval.py --output results.json
    python eval.py --grounded-only          # only questions with ground_truth filled in
    python eval.py --include-feedback       # also pull questions from feedback.jsonl
"""

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    AnswerCorrectness,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from src.retriever import ask, retrieve

load_dotenv()

EVAL_QUESTIONS_FILE = Path("eval_questions.json")
FEEDBACK_LOG = Path("feedback.jsonl")


def _load_eval_questions() -> list[dict]:
    if not EVAL_QUESTIONS_FILE.exists():
        return []
    with EVAL_QUESTIONS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _load_feedback_questions() -> list[dict]:
    if not FEEDBACK_LOG.exists():
        return []
    seen: set[str] = set()
    questions = []
    with FEEDBACK_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                q = entry.get("question", "").strip()
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    questions.append({"question": q, "ground_truth": ""})
            except json.JSONDecodeError:
                continue
    return questions


def _build_sample(question: str, ground_truth: str, api_key: str) -> dict | None:
    try:
        result = ask(question)
    except Exception as e:
        print(f"  [skip] '{question[:60]}' — ask() failed: {e}")
        return None

    try:
        chunks = retrieve(
            result.get("search_query", question),
            api_key=api_key,
            category=result.get("category"),
            level_group=result.get("level_group"),
        )
        contexts = [c["text"] for c in chunks]
    except Exception as e:
        print(f"  [skip] '{question[:60]}' — retrieve() failed: {e}")
        return None

    return {
        "user_input": question,
        "response": result["answer"],
        "retrieved_contexts": contexts,
        "reference": ground_truth or None,
    }


def _run_eval(samples: list[dict], metrics: list, llm, label: str) -> dict:
    dataset = Dataset.from_dict(
        {
            "user_input": [s["user_input"] for s in samples],
            "response": [s["response"] for s in samples],
            "retrieved_contexts": [s["retrieved_contexts"] for s in samples],
            "reference": [s.get("reference") for s in samples],
        }
    )

    print(f"\nRunning {label} on {len(dataset)} questions...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm)

    df = result.to_pandas()
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference",
                 "question", "answer", "contexts", "ground_truth"}
    metric_cols = [c for c in df.columns if c not in skip_cols]

    print(f"\n--- {label} Results ---")
    scores: dict[str, float] = {}
    for col in metric_cols:
        score = float(df[col].mean())
        scores[col] = score
        print(f"  {col:<50} {score:.3f}")

    return {"scores": scores, "details": df.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the cheer rules pipeline.")
    parser.add_argument("--output", type=Path, default=None, help="Save all results to this JSON file.")
    parser.add_argument("--grounded-only", action="store_true",
                        help="Only evaluate questions that have ground_truth filled in.")
    parser.add_argument("--include-feedback", action="store_true",
                        help="Add questions from feedback.jsonl (no ground_truth) to the eval set.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Add it to your .env file.")
        return

    # Collect questions
    questions = _load_eval_questions()

    if args.include_feedback:
        seen = {q["question"].lower() for q in questions}
        for q in _load_feedback_questions():
            if q["question"].lower() not in seen:
                questions.append(q)
                seen.add(q["question"].lower())

    if not questions:
        print("No questions found. Populate eval_questions.json or use --include-feedback.")
        return

    if args.grounded_only:
        questions = [q for q in questions if q.get("ground_truth", "").strip()]
        if not questions:
            print("No questions with ground_truth found. Fill in eval_questions.json first.")
            return

    # Gemini judge LLM
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    )

    # Build samples by running each question through the pipeline
    print(f"\nBuilding dataset from {len(questions)} questions...")
    samples: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question'][:70]}")
        sample = _build_sample(q["question"], q.get("ground_truth", ""), api_key)
        if sample is not None:
            samples.append(sample)

    if not samples:
        print("All questions failed — nothing to evaluate.")
        return

    all_results: dict = {}

    # Reference-free eval — runs on every sample
    all_results["reference_free"] = _run_eval(
        samples,
        metrics=[Faithfulness(), ContextPrecisionWithoutReference()],
        llm=llm,
        label="Reference-Free Metrics",
    )

    # Reference eval — only samples where ground_truth was provided
    grounded = [s for s in samples if s.get("reference")]
    if grounded:
        all_results["reference"] = _run_eval(
            grounded,
            metrics=[ContextRecall(), AnswerCorrectness()],
            llm=llm,
            label="Reference Metrics",
        )
    else:
        print("\nNo grounded samples found — skipping ContextRecall and AnswerCorrectness.")
        print("Fill in 'ground_truth' values in eval_questions.json to enable these metrics.")

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
