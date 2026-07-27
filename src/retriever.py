"""
Retrieval-Augmented Generation for cheer skill legality.

Run directly to test:
    python src/retriever.py "Is a basket toss legal at Level 4?"
"""

import os
import re
import sys
from pathlib import Path

import chromadb
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "cheer_rules"
EMBED_MODEL = "gemini-embedding-2"
CHAT_MODEL = "gemini-2.5-flash"
N_RESULTS = 8  # number of rulebook chunks to retrieve per query

_CATEGORIES = frozenset({"TUMBLING", "STUNTS", "PYRAMIDS", "DISMOUNTS", "TOSSES"})


class _GeminiEmbedder:
    """Custom ChromaDB embedding function using the google-genai SDK."""
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def __call__(self, input):  # noqa: A002
        return [
            self._client.models.embed_content(model=EMBED_MODEL, contents=text).embeddings[0].values
            for text in input
        ]

_LEVEL7_RE = re.compile(r"\blevel\s*7\b", re.IGNORECASE)
_LEVEL16_RE = re.compile(r"\blevel\s*[1-6]\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"\b(what is|what are|define|definition of|meaning of|explain)\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are an expert cheer rules official with deep knowledge of IASF and USASF rulebooks.

TWO REFERENCE SOURCES — understand the difference:
- IASF rulebook (iasf_rules_cheer_25-28_final.pdf): the authoritative legal document. Use this for ALL legality questions, specific conditions, restrictions, and rule citations.
- Cheer Canada At Level Skill List (skills chart PDF): a non-exhaustive quick reference listing which skills are appropriate at each level. Use this to answer "what skills can I do at level X" questions. Do NOT use it as a source for legality conditions or rule citations — it does not contain those.

CHUNK HEADER FORMAT — use this to identify which level content belongs to:
- "CATEGORY — SECTION" headers come from the Levels 1–6 combined table. Each rule inside is prefixed "LEVEL N:" so you can tell exactly which level it applies to.
- "CATEGORY LEVEL 7 — SECTION" headers come from the separate Level 7 table. ALL content under that header applies to Level 7 specifically.
- If a skill only appears under a "CATEGORY LEVEL 7" header and is absent from the Level 1–6 table, it is first allowed at Level 7.

SKILL PROGRESSION RULES — apply these to every answer:
- Levels run from 1 (most restrictive) to 7 (least restrictive).
- If a skill first appears as allowed at Level X, it is NOT allowed below Level X.
- If a skill is not mentioned for a level, it is NOT allowed at that level.
- If a skill is listed for Level X and above, state clearly: "First allowed at Level X."
- CARRY-FORWARD rule applies to PROHIBITED SKILLS ONLY: if a skill is outright prohibited at Level N (e.g. "NOT allowed", "NO helicopters"), that prohibition carries forward to lower levels. It does NOT automatically apply to higher levels — higher levels may add permissions not present at lower levels.
- LEVEL-SPECIFIC CONDITIONS (starting height, twist limits, number of catchers, etc.) are defined per level in that level's column. If a restrictive condition appears at Level N but is NOT repeated at Level N+1, it does NOT carry forward — higher levels gain freedom. Only cite the conditions listed for the SPECIFIC level being asked about.

BASIC MECHANICS vs SPECIFIC SKILLS — critical distinction:
- The "if not mentioned, not allowed" rule applies to SPECIFIC SKILLS (twisting, inversions, release moves, specific body positions like liberties, helicopters, etc.).
- It does NOT apply to basic stunt mechanics: controlled loading into a stunt, controlled lowering/dismounting back to the ground, or spotting. These are legal at all levels unless a rule explicitly prohibits them.
- A "sponge down" or "sponge load" is a controlled loading/unloading motion by the bases — it is NOT a drop. A drop is an uncontrolled fall onto a body part (knee, seat, front, back). Do not conflate these.

Answer based on the TYPE of question:

DEFINITION questions ("what is X", "define X", "what does X mean"):
- The definition may be split across multiple excerpts from the same page — combine them ALL in the order they appear.
- Include EVERY sentence and EVERY bullet point from the definition. Do not omit, paraphrase, or summarize any part.
- Reproduce the full definition verbatim, including opening paragraphs, all bullet points, and any closing sentences.
- Do NOT use the LEGAL/ILLEGAL format for definitions.

LEGALITY questions ("can I do X", "is X legal", "what level is X allowed"):

SKILL LISTING questions ("what can I do at level X", "what's allowed at level X", "what [category] skills are allowed at level X"):
- List ONLY the specific skills from the Level X column of the relevant rulebook table.
- Do NOT list Section A (General) rules — those apply uniformly to all levels and do not answer "what skills are available at this level."
- Include any level-specific or division-specific notes exactly as written (e.g. "NO tosses allowed in U8 Division").
- Format: simply list the skills and level-specific notes verbatim from the rulebook, one per line. No LEGAL/ILLEGAL header needed.

SINGLE SKILL questions (one move, no sequence):
- State LEGAL / ILLEGAL / CANNOT DETERMINE on its own line.
- If LEGAL: state "First allowed at Level X." then list the exact conditions from the rulebook under a heading like "Conditions at Level X:" — these are stipulations for performing the skill legally, not violations.
- If ILLEGAL: state the rule being violated with exact wording and its citation code.
- Do NOT use the "Step N:" format for single skills.

SEQUENCE questions (multiple steps described):
- Evaluate each step INDIVIDUALLY using "Step N (description): LEGAL/ILLEGAL — [rule and citation]".
- Clearly state which specific step is legal and which is not — do NOT label the whole sequence illegal just because one step violates a rule.
- Only say CANNOT DETERMINE if the excerpts genuinely do not mention the skill at all.

FOR ALL LEGALITY ANSWERS:
- List every condition using EXACT wording from the rulebook — do not paraphrase, interpret, or merge clauses.
- Treat each sentence, semicolon-separated clause, and exception line as a SEPARATE bullet point.
- Do NOT combine separate clauses.
- Only cite rules that DIRECTLY address the specific skill, height, twist, or technique being questioned. Do NOT cite general methodology statements such as "All skills allowed at a particular level encompass all skills allowed in the preceding level" — those are background principles, not the rule being violated.
- Only apply rules from the same category as the skill being evaluated (e.g. for a stunt sequence, only use rules from STUNTS chunks — do not apply Tosses or Pyramids rules to a stunt question even if those chunks appear in the excerpts).
- Spotter requirements are defined in the STUNTS table section A (Spotters) — cite them as [Stunts A{level}], NOT as General Rules, even if the general rules also mention spotters.
- When a question is about performing WITHOUT a spotter, the relevant rule is the STUNTS — A. SPOTTER entry that states AT WHAT HEIGHT a spotter is mandatory (e.g. "Required for prep level and above; Floor stunts"). Quote that level-specific height requirement verbatim from the excerpt. Do NOT quote the general definition of what a spotter is or who qualifies as one ("must be your own team's members", "stands to the side", etc.) — those describe the spotter role, not the height requirement being violated.
- When a safety requirement is MISSING (e.g. "without a spotter", "without catchers"), every step that requires that role is ILLEGAL from the very first step. Do not mark step 1 as LEGAL just because the action itself is a legal skill — if the required safety role is absent, the step is ILLEGAL regardless. Mark ALL affected steps as ILLEGAL and cite the safety rule once at the top.
- A step is ILLEGAL if it results in the stunt being AT or ABOVE the height that requires a spotter. Steps that remain below that threshold are legal even without a spotter. Example: a step that loads only to waist level is legal without a spotter; a step that loads TO prep level (reaching that height) is illegal without a spotter because the stunt ends at the height requiring one.

RULE CITATION FORMAT:
- Citation codes [Table SectionLevel] are for ILLEGAL rules ONLY — they identify which rule is being violated.
- Do NOT add citation codes to conditions listed for a LEGAL skill. When a skill is legal, just list the conditions as plain text — no brackets.
- For ILLEGAL rules: use the cite-as tag from the excerpt label (e.g. "cite-as: Stunts A" + Level 1 → [Stunts A1]). NEVER invent a citation from memory. NEVER use numbered general rules like "General Rules 12". If an excerpt has no cite-as tag, do not add a citation code.
- Always place the citation code on the same line as the rule it refers to.

LEVEL 7 — include ONLY for "what level can I do X" questions (spanning all levels):
- In a level-by-level breakdown, include LEVEL 7 from the separate "CATEGORY LEVEL 7" table.
- Do NOT include Level 7 content when the question asks about a SPECIFIC level (e.g. "what can I do at Level 3", "is X legal at Level 2"). Answer only for the level that was asked."""


_CLASSIFY_PROMPT = (
    "You are categorizing a cheer skill question into one of the five main rulebook tables.\n"
    "TUMBLING: floor skills — back handspring, layout, tuck, full, round-off, pass, aerial\n"
    "STUNTS: single elevated person on bases — extension, liberty, heel stretch, scorpion, load in, "
    "sponge up/down (loading/unloading motion into a stunt), push to extension, prep level, stunt height, "
    "release moves (helicopter, rewind, tic-toc, salto), performing a stunt without a spotter (spotter requirements live in STUNTS section A)\n"
    "PYRAMIDS: multiple CONNECTED top persons — pyramid structures where one top person physically braces another top person\n"
    "DISMOUNTS: coming down from stunts or pyramids — straight cradle, twisting dismount, pop-off\n"
    "TOSSES: person thrown INTO THE AIR from ground by bases — basket toss, sponge TOSS, elevator toss\n"
    "CRITICAL DISTINCTIONS:\n"
    "- 'sponge load' / 'sponge up' / 'sponge down' as part of a stunt sequence = controlled loading/lowering motion, NOT a drop and NOT a toss. Treat as basic stunt mechanics.\n"
    "- 'sponge toss' = TOSSES (person is thrown into the air)\n"
    "- 'helicopter' = STUNTS (it is a stunt release move — the glossary says 'tossed' but it is classified under STUNTS release moves, NOT TOSSES)\n"
    "- 'spotter' (safety person beside stunt) = STUNTS, not PYRAMIDS\n"
    "- Any question about performing a stunt 'without a spotter' = STUNTS (spotter rules are in STUNTS section A)\n"
    "- 'bracer' (top person physically connected to another top person) = PYRAMIDS\n"
    "Return exactly one word: TUMBLING, STUNTS, PYRAMIDS, DISMOUNTS, TOSSES, or NONE."
)


def _detect_level_group(query: str) -> str | None:
    """Return 'level7' if the query is about Level 7, 'standard' if Levels 1-6, else None."""
    if _LEVEL7_RE.search(query):
        return "level7"
    if _LEVEL16_RE.search(query):
        return "standard"
    return None


def _classify_category(query: str, client: genai.Client) -> str | None:
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=query,
        config=genai_types.GenerateContentConfig(
            system_instruction=_CLASSIFY_PROMPT,
            temperature=0,
            max_output_tokens=10,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    cat = (resp.text or "").strip().upper()
    return cat if cat in _CATEGORIES else None


def _get_collection(api_key: str):
    embedder = _GeminiEmbedder(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return chroma.get_collection(name=COLLECTION_NAME, embedding_function=embedder)


def retrieve(
    query: str,
    api_key: str,
    n_results: int = N_RESULTS,
    category: str | None = None,
    level_group: str | None = None,
) -> list[dict]:
    """Return the top-n most relevant rulebook chunks for a query."""
    collection = _get_collection(api_key)

    conditions = []
    if category:
        conditions.append({"category": {"$eq": category}})
    if level_group:
        conditions.append({"level_group": {"$eq": level_group}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    # Try the tightest filter first, then progressively loosen if nothing comes back.
    # This prevents hallucination when the category/level_group filter is too strict.
    filter_attempts: list[dict | None] = []
    if where is not None:
        filter_attempts.append(where)
    if category and level_group:
        filter_attempts.append({"category": {"$eq": category}})  # drop level_group
    filter_attempts.append(None)  # fully unfiltered fallback

    results = None
    for attempt in filter_attempts:
        results = collection.query(query_texts=[query], n_results=n_results, where=attempt)
        if results["documents"][0]:
            break

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "section": meta.get("section", 0),
            "chunk_idx": meta.get("chunk", 0),
            "score": round(1 - distance, 3),  # cosine similarity (higher = more relevant)
        })
    return chunks


_CITE_AS_RE = re.compile(
    r"^(TUMBLING|STUNTS|PYRAMIDS|DISMOUNTS|TOSSES)(?:\s+LEVEL\s*7)?(?:\s+.*?)?—\s*([A-Z])\.",
    re.IGNORECASE,
)


def _derive_cite_as(text: str) -> str | None:
    """Return 'Table Section' from a chunk's first line header, e.g. 'Stunts A'."""
    first_line = text.split("\n")[0]
    m = _CITE_AS_RE.match(first_line)
    if not m:
        return None
    table = m.group(1).capitalize()
    section = m.group(2).upper()
    return f"{table} {section}"


_REWRITE_PROMPT = (
    "You are a cheer rules terminology expert. Rewrite the following question using "
    "official IASF/USASF rulebook language so it matches the rulebook's exact wording. "
    "Use terms like: 'two leg extended', 'single leg extended', 'liberty', 'prep level', "
    "'extended stunt', 'release move', 'inversion', 'top person', 'base'. "
    "When a question describes a multi-step sequence, identify the SPECIFIC step that could "
    "violate a rule and focus the rewritten query on that step and its rule category "
    "(e.g. stunt height, twisting, inversion). "
    "IMPORTANT DISTINCTIONS — do not confuse these:\n"
    "- 'spotter': a safety person standing beside the stunt to protect the top person. "
    "Keep the word 'spotter' — do NOT convert it to 'bracer'.\n"
    "- 'bracer': a second top person physically connected to another top person in a PYRAMID "
    "for stability. Only use 'bracer' when the question is about pyramid connections.\n"
    "Return only the rewritten query — no explanation."
)


def _rewrite_query(query: str, client: genai.Client) -> str:
    """Rephrase the user's question into official rulebook terminology for better retrieval."""
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=query,
        config=genai_types.GenerateContentConfig(
            system_instruction=_REWRITE_PROMPT,
            temperature=0,
            max_output_tokens=200,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (resp.text or query).strip()


def ask(query: str) -> dict:
    """Retrieve relevant chunks and ask Gemini to answer the legality question."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Add it to .env")
    client = genai.Client(api_key=api_key)

    is_definition = bool(_DEFINITION_RE.search(query))

    if is_definition:
        # Definitions: use the original query verbatim — rewriting "spotter" to "bracer"
        # or similar breaks glossary retrieval. Also skip category/level filters since
        # glossary entries span all categories and levels.
        search_query = query
        category = None
        level_group = None
        n_results = 15
    else:
        search_query = _rewrite_query(query, client)
        category = _classify_category(query, client)
        level_group = _detect_level_group(query)
        n_results = N_RESULTS

        # Keyword override: classifier often misses "without a spotter" stunt questions.
        # Spotter rules live exclusively in STUNTS section A — force category so the
        # unfiltered fallback doesn't surface General Rules instead.
        if category is None and re.search(r"without\s+a?\s*spotter", query, re.IGNORECASE):
            if not re.search(r"\b(pyramid|tumbling|toss)\b", query, re.IGNORECASE):
                category = "STUNTS"

    chunks = retrieve(search_query, api_key=api_key, n_results=n_results, category=category, level_group=level_group)

    if is_definition and chunks:
        # Glossary entries are self-contained on one page. Focus on all chunks from the
        # top-scoring page only — this removes cross-page noise that distracts GPT and
        # causes it to omit bullet points. Present as plain sequential text so GPT
        # reads one continuous definition rather than N separate labelled excerpts.
        top_page = max(chunks, key=lambda c: c["score"])["page"]
        definition_chunks = sorted(
            [c for c in chunks if c["page"] == top_page],
            key=lambda c: (c["section"], c["chunk_idx"]),
        )
        context = "\n\n".join(c["text"] for c in definition_chunks)
    else:
        def _chunk_label(c: dict) -> str:
            cite = _derive_cite_as(c["text"])
            cite_tag = f" | cite-as: {cite}" if cite else ""
            return f"[{c['source']}, p.{c['page']} | relevance: {c['score']}{cite_tag}]\n{c['text']}"

        context = "\n\n---\n\n".join(_chunk_label(c) for c in chunks)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"RULEBOOK EXCERPTS:\n{context}\n\nQUESTION: {query}",
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    return {
        "question": query,
        "search_query": search_query,
        "category": category,
        "level_group": level_group,
        "answer": response.text,
        "sources": [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks],
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What skills are restricted at Level 1?"
    print(f"\nQuestion: {query}\n")
    result = ask(query)
    print(f"(Searched rulebook for: {result['search_query']})")
    if result.get("category"):
        parts = [result["category"]]
        if result.get("level_group"):
            parts.append(result["level_group"])
        print(f"(Category: {' / '.join(parts)})")
    print()
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  • {s['source']}, page {s['page']} (relevance: {s['score']})")
