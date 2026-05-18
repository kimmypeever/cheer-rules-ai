"""
Show what category/level_group metadata is stored in ChromaDB.
Run: python debug_db.py
"""
import chromadb

col = chromadb.PersistentClient(path="vectorstore").get_collection("cheer_rules")
print(f"Total chunks: {col.count()}\n")

for cat in ["TOSSES", "STUNTS", "PYRAMIDS", "TUMBLING", "DISMOUNTS"]:
    r = col.get(where={"category": {"$eq": cat}}, include=["metadatas"])
    if r["ids"]:
        level_groups = set(m.get("level_group", "untagged") for m in r["metadatas"])
        pages = sorted(set(m["page"] for m in r["metadatas"]))
        print(f"{cat}: {len(r['ids'])} chunks | level_groups={level_groups} | pages={pages}")
    else:
        print(f"{cat}: 0 chunks (not tagged)")

untagged = col.get(include=["metadatas"])
no_cat = sum(1 for m in untagged["metadatas"] if "category" not in m)
print(f"\nChunks with no category tag: {no_cat}")
