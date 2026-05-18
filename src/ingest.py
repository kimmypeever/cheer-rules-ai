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
import pdfplumber
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


load_dotenv()

RULEBOOKS_DIR = Path("data/rulebooks")
VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "cheer_rules"

CHUNK_SIZE = 900      # characters; ~200-220 tokens — keeps rule clauses intact
CHUNK_OVERLAP = 120   # carry context across chunk boundaries

# Patterns that signal a hard rule boundary — prefer splitting before these.
# NOTE: "LEVEL\s+\d" is intentionally excluded here because our table formatter
# already uses "LEVEL N: …" as content prefixes within grouped blocks, and splitting
# there would separate the level label from its category context.
_RULE_BOUNDARY = re.compile(
    r"(?m)(?="
    r"(?:\d+\.\d[\d.]*\s)"      # 4.1, 4.1.1, 4.1.1.1 …
    r"|(?:Article\s+\d)"         # Article 5
    r"|(?:Section\s+\d)"         # Section 3
    r")"
)


_SECTION_LETTER = re.compile(r"^[A-Z]\.\s")  # matches "A. GENERAL", "B. STRUCTURES …"

_CATEGORIES = frozenset({"TUMBLING", "STUNTS", "PYRAMIDS", "DISMOUNTS", "TOSSES"})


def _detect_category(chunk: str) -> str | None:
    """Return the main rulebook category if the chunk's first line contains one."""
    first_line = chunk.split("\n")[0].upper()
    for cat in _CATEGORIES:
        # Matches "STUNTS — ..." (IASF format) and "LEVEL 2 - STUNTS" (skills list format)
        if first_line.startswith(cat) or f" {cat}" in first_line or f"-{cat}" in first_line:
            return cat
    return None


def _format_table(table: list[list]) -> str:
    """
    Format a pdfplumber table for embedding.

    IASF tables look like:
      Row 0: ['STUNTS', '', '', ...]          ← category name (merged cell)
      Row 1: ['LEVEL 1', 'LEVEL 2', ...]      ← level column headers
      Row 2: ['A. GENERAL', '', '', ...]       ← section header (merged)
      Row 3: [per-level content …]             ← actual rules per level

    Strategy: group every section's per-level content together with its
    category + section header into one text block.  This ensures the chunk
    always contains both "PYRAMIDS — B. STRUCTURES" and "LEVEL 1: …" so
    retrieval never misses the category context.

    For single-column tables (e.g. Level 7 rules), falls back to plain text.
    """
    rows = [
        [str(cell or "").strip() for cell in row]
        for row in table
        if any(cell for cell in row)
    ]
    if not rows:
        return ""

    # Find the row that contains the LEVEL headers
    level_row_idx = next(
        (i for i, row in enumerate(rows) if any("LEVEL" in c.upper() for c in row if c)),
        None,
    )

    if level_row_idx is None:
        # Single-column or non-level table — emit as readable prose
        return "\n".join(c for row in rows for c in row if c)

    # Rows before the level header → category name (e.g. "STUNTS", "PYRAMIDS")
    category = " ".join(
        c for row in rows[:level_row_idx] for c in row if c
    )

    headers = rows[level_row_idx]

    # For single-level tables (Level 7), fold the level label into the category so
    # every chunk explicitly mentions e.g. "TUMBLING LEVEL 7" for retrieval.
    unique_levels = [h for h in headers if h]
    if len(unique_levels) == 1 and unique_levels[0] not in category:
        category = f"{category} {unique_levels[0]}".strip()

    # Walk content rows, grouping by section.
    # Each section becomes one text block:  "CATEGORY — SECTION\nLEVEL 1: …\nLEVEL 2: …"
    blocks: list[str] = []
    current_section: str = ""
    level_lines: list[str] = []
    general_lines: list[str] = []

    def _flush():
        if not level_lines and not general_lines:
            return
        ctx_parts = [p for p in [category, current_section] if p]
        header_line = " — ".join(ctx_parts)
        block_lines = ([header_line] if header_line else []) + general_lines + level_lines
        blocks.append("\n".join(block_lines))
        level_lines.clear()
        general_lines.clear()

    for row in rows[level_row_idx + 1:]:
        non_empty = [(h, c) for h, c in zip(headers, row) if c]
        if not non_empty:
            continue
        if len(non_empty) == 1:
            cell = non_empty[0][1]
            if _SECTION_LETTER.match(cell):
                # New lettered section (A. / B. / C. …) — flush previous section first
                _flush()
                current_section = cell
            else:
                # General rule that applies to all levels within this section
                general_lines.append(cell)
        else:
            # Per-level rule content
            for header, cell in non_empty:
                level_lines.append(f"{header}: {cell}")

    _flush()
    return "\n\n".join(blocks)


def _split_at_category_boundaries(table: list[list]) -> list[list[list]]:
    """
    pdfplumber sometimes merges two adjacent tables (e.g. TUMBLING + STUNTS on page 18)
    into a single table object.  This function detects rows whose sole non-empty cell is a
    known category name and uses them as split points, returning one sub-table per category.
    """
    sub_tables: list[list[list]] = []
    current: list[list] = []

    for row in table:
        non_empty = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        is_new_category = (
            len(non_empty) == 1
            and non_empty[0].upper() in _CATEGORIES
            and current  # don't split on the very first row
        )
        if is_new_category:
            sub_tables.append(current)
            current = [row]
        else:
            current.append(row)

    if current:
        sub_tables.append(current)

    return sub_tables if sub_tables else [table]


def _extract_pages(pdf_path: Path) -> list[dict]:
    # Returns one entry per TABLE (not per page) so that pages containing multiple
    # tables (e.g. DISMOUNTS + TOSSES on the same page) each get their own category
    # metadata rather than being tagged with only the first table's category.
    sections: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"  Extracting page {page_num}/{total} …", end="\r")
            has_table = False
            section_idx = 0
            try:
                for table in page.extract_tables({"text_tolerance": 5}):
                    for sub_table in _split_at_category_boundaries(table):
                        md = _format_table(sub_table)
                        if not md:
                            continue
                        has_table = True
                        section: dict = {
                            "text": md,
                            "source": pdf_path.name,
                            "page": page_num,
                            "section": section_idx,
                        }
                        section_idx += 1
                        first_line = md.split("\n")[0].upper()
                        for cat in _CATEGORIES:
                            if first_line.startswith(cat):
                                section["category"] = cat
                                section["level_group"] = "level7" if "LEVEL 7" in first_line else "standard"
                                break
                        sections.append(section)
            except Exception as exc:
                print(f"\n  Page {page_num}: table extraction failed ({exc}), using plain text.")

            # Only add raw text for non-table pages to avoid duplicate/misleading chunks.
            if not has_table:
                # Split the page at its horizontal midpoint and extract each column
                # separately so two-column pages (e.g. the glossary) read left column
                # then right column rather than row-by-row across both columns.
                mid = page.width / 2
                left = (page.crop((0, 0, mid, page.height)).extract_text() or "").strip()
                right = (page.crop((mid, 0, page.width, page.height)).extract_text() or "").strip()
                if left and right:
                    text = left + "\n\n" + right  # two-column: left then right
                else:
                    text = (page.extract_text() or "").strip()  # single-column fallback
                if text:
                    sections.append({"text": text, "source": pdf_path.name, "page": page_num, "section": 0})

    print()  # newline after progress line
    return sections


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
            if end >= len(buf):
                start = len(buf)
            elif end - CHUNK_OVERLAP > start:
                start = end - CHUNK_OVERLAP  # normal overlap
            else:
                start = end  # separator was too close to start; skip overlap to guarantee progress

    for seg in segments:
        if len(buf) + len(seg) <= CHUNK_SIZE:
            buf = buf + (" " if buf else "") + seg
        else:
            flush(buf)
            buf = seg
    flush(buf)

    return [c for c in chunks if c]


def _chunk_id(source: str, page: int, section: int, chunk_idx: int) -> str:
    raw = f"{source}:p{page}:s{section}:c{chunk_idx}"
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
                cid = _chunk_id(page_data["source"], page_data["page"], page_data.get("section", 0), chunk_idx)
                if cid in existing_ids:
                    continue
                ids.append(cid)
                docs.append(chunk)
                meta = {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk": chunk_idx,
                }
                # Chunk-level detection wins; page-level is the fallback for continuation chunks
                # that don't start with the category header after a split.
                cat = _detect_category(chunk) or page_data.get("category")
                if cat:
                    meta["category"] = cat
                if "level_group" in page_data:
                    meta["level_group"] = page_data["level_group"]
                metas.append(meta)

        if not ids:
            print("  → Already fully indexed, skipping.")
            continue

        print(f"  {len(ids)} chunks ready — sending to OpenAI for embedding …")
        batch = 100
        for i in range(0, len(ids), batch):
            batch_num = i // batch + 1
            batch_total = (len(ids) + 99) // 100
            print(f"  Batch {batch_num}/{batch_total} ({min(i + batch, len(ids))} chunks) …")
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
