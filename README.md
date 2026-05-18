# Cheer Rules AI

A Retrieval-Augmented Generation (RAG) assistant that answers questions about competitive cheerleading skill legality using the IASF/USASF rulebooks.

Ask it things like:
- *"Is a basket toss legal at Level 3?"*
- *"What level can I do a helicopter?"*
- *"Can I load to prep without a spotter at Level 1?"*
- *"What tumbling is allowed at Level 2?"*

## How it works

1. **Ingest** — rulebook PDFs are parsed, chunked, and embedded into a local ChromaDB vector store using OpenAI embeddings.
2. **Retrieve** — each question is classified by skill category (Stunts, Tosses, Tumbling, etc.) and the most relevant rulebook chunks are retrieved.
3. **Answer** — GPT-4o-mini generates a structured answer grounded in the retrieved excerpts, with source page citations.

The system handles six question types automatically: skill listing, single-skill legality, level-by-level breakdowns, multi-step sequences, missing safety role (e.g. "without a spotter"), and glossary definitions.

## Project structure

```
cheer-rules-ai/
├── data/
│   └── rulebooks/          # Drop rulebook PDFs here before ingesting
├── vectorstore/            # ChromaDB persisted embeddings (auto-created)
├── src/
│   ├── ingest.py           # PDF → chunks → ChromaDB
│   └── retriever.py        # RAG query pipeline + GPT answer generation
├── api.py                  # FastAPI wrapper (one POST /ask endpoint)
├── frontend/               # Next.js + shadcn/ui chat interface
└── prompt_templates.md     # Reference for question types and response formats
```

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/cheer-rules-ai.git
cd cheer-rules-ai
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source .venv/bin/activate`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### 4. Add rulebook PDFs

Place your rulebook PDFs in `data/rulebooks/`. The project uses:
- `iasf_rules_cheer_25-28_final.pdf` — the authoritative IASF rulebook (legality, conditions, citations)
- `All Star & Prep Level Appropriate Skills Chart (1).pdf` — Cheer Canada skill listing by level

### 5. Ingest the PDFs

```bash
python src/ingest.py
```

This parses the PDFs, chunks them, embeds them with OpenAI, and saves them to the local vector store. Only new chunks are added on subsequent runs.

To wipe and re-ingest everything from scratch:

```bash
python src/ingest.py --reset
```

### 6. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running

You need two terminals running simultaneously.

**Terminal 1 — API server** (from the project root, with venv active):

```bash
uvicorn api:app --reload
```

The API runs at `http://localhost:8000`. You can explore it at `http://localhost:8000/docs`.

**Terminal 2 — Frontend** (from the `frontend/` directory):

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

## Testing the backend without the frontend

You can query the retriever directly from the command line:

```bash
python src/retriever.py "Is a basket toss legal at Level 3?"
```

Or call the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What level can I do a helicopter?"}'
```

## API reference

### `POST /ask`

**Request:**
```json
{ "query": "Is a basket toss legal at Level 3?" }
```

**Response:**
```json
{
  "question": "Is a basket toss legal at Level 3?",
  "search_query": "basket toss legality Level 3 twisting tricks",
  "category": "TOSSES",
  "level_group": "standard",
  "answer": "LEGAL — First allowed at Level 2.\n\nConditions at Level 3:\n...",
  "sources": [
    { "source": "iasf_rules_cheer_25-28_final.pdf", "page": 21, "score": 0.847 }
  ]
}
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required. Used for embeddings and GPT-4o-mini completions. |

Frontend environment (`frontend/.env.local`):

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | URL of the FastAPI backend. |
