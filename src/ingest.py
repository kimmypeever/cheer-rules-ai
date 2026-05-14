"""
Load rulebook PDFs → chunk → embed with OpenAI → persist to ChromaDB.

Run directly:
    python src/ingest.py                  # ingest any new PDFs
    python src/ingest.py --reset          # wipe collection and re-ingest everything
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

RULEBOOKS_DIR = Path("data/rulebooks")
VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "cheer_rules"

CHUNK_SIZE = 900      # characters; ~200-220 tokens — keeps rule clauses intact
CHUNK_OVERLAP = 120   # carry context across chunk boundaries

# Patterns that signal a hard rule boundary — prefer splitting before these
_RULE_BOUNDARY = re.compile(
    r"(?m)(?="
    r"(?:\d+\.\d[\d.]*\s)"      # 4.1, 4.1.1, 4.1.1.1 …
    r"|(?:Article\s+\d)"         # Article 5
    r"|(?:Section\s+\d)"         # Section 3
    r"|(?:LEVEL\s+\d)"           # LEVEL 4 (USASF style)
    r"|(?:[A-Z]{2,}\s*:\s)"      # RESTRICTION: / NOTE: / EXCEPTION:
    r")"
)


def _extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": pdf_path.name, "page": page_num})
    return pages


def _split_into_chunks(text: str) -> list[str]:
    """
    Split text into overlapping chunks.  Tries to break at rule-boundary
    patterns first, then double-newlines, then sentences, then spaces.
    """
    segments: list[str] = [s for s in _RULE_BOUNDARY.split(text) if s.strip()]
    if not segments:
        segments = [text]

    chunks: list[str] = []
    buf = ""

    def flush(buf: str) -> None:
        buf = buf.strip()
        if not buf:
            return
        if len(buf) <= CHUNK_SIZE:
            chunks.append(buf)
            return
        fallback_seps = ["\n\n", "\n", ". ", " "]
        start = 0
        while start < len(buf):
            end = min(start + CHUNK_SIZE, len(buf))
            if end < len(buf):
                for sep in fallback_seps:
                    pos = buf.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunks.append(buf[start:end].strip())
            start = end - CHUNK_OVERLAP if end < len(buf) else len(buf)

    for seg in segments:
        if len(buf) + len(seg) <= CHUNK_SIZE:
            buf = buf + (" " if buf else "") + seg
        else:
            flush(buf)
            buf = seg
    flush(buf)

    return [c for c in chunks if c]


def _chunk_id(source: str, page: int, chunk_idx: int) -> str:
    raw = f"{source}:p{page}:c{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest(rulebooks_dir: Path = RULEBOOKS_DIR, reset: bool = False) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    pdf_files = sorted(rulebooks_dir.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDFs found in {rulebooks_dir}/. Drop rulebook PDFs there and re-run.")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' wiped.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])

    total_new = 0
    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name} …")
        pages = _extract_pages(pdf_path)
        if not pages:
            print("  → No extractable text, skipping (scanned PDF?).")
            continue

        ids, docs, metas = [], [], []
        for page_data in pages:
            for chunk_idx, chunk in enumerate(_split_into_chunks(page_data["text"])):
                cid = _chunk_id(page_data["source"], page_data["page"], chunk_idx)
                if cid in existing_ids:
                    continue
                ids.append(cid)
                docs.append(chunk)
                metas.append({
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk": chunk_idx,
                })

        if not ids:
            print("  → Already fully indexed, skipping.")
            continue

        batch = 100
        for i in range(0, len(ids), batch):
            collection.add(
                ids=ids[i : i + batch],
                documents=docs[i : i + batch],
                metadatas=metas[i : i + batch],
            )

        total_new += len(ids)
        print(f"  → {len(ids)} new chunks added (across {len(pages)} pages).")

    print(
        f"\nIngestion complete. {total_new} new chunks added. "
        f"Collection '{COLLECTION_NAME}' now holds {collection.count()} total chunks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest cheer rulebook PDFs into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before ingesting (full re-embed).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=RULEBOOKS_DIR,
        help=f"Directory containing PDF rulebooks (default: {RULEBOOKS_DIR}).",
    )
    args = parser.parse_args()
    ingest(rulebooks_dir=args.dir, reset=args.reset)
