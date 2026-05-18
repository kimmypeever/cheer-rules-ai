"""
Debug: what chunks are retrieved for the spotter query?
Run: python debug_spotter.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
)
col = chromadb.PersistentClient(path="vectorstore").get_collection(
    "cheer_rules", embedding_function=openai_ef
)

query = "stunt without a spotter spotter requirements level 1"

print("=== STUNTS filter ===")
r = col.query(
    query_texts=[query],
    n_results=10,
    where={"category": {"$eq": "STUNTS"}},
)
if not r["documents"][0]:
    print("  NO RESULTS with STUNTS filter")
else:
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"\n[p.{meta['page']} | cat={meta.get('category','?')} | lg={meta.get('level_group','?')} | score={round(1-dist,3)}]")
        print(doc[:300])

print("\n\n=== Unfiltered ===")
r2 = col.query(query_texts=[query], n_results=10)
for doc, meta, dist in zip(r2["documents"][0], r2["metadatas"][0], r2["distances"][0]):
    print(f"\n[p.{meta['page']} | cat={meta.get('category','?')} | lg={meta.get('level_group','?')} | score={round(1-dist,3)}]")
    print(doc[:300])
