import chromadb, os, sys
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

query = " ".join(sys.argv[1:]) or "helicopter"

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)
col = chromadb.PersistentClient(path="vectorstore").get_collection(
    "cheer_rules", embedding_function=ef
)

results = col.query(query_texts=[query], n_results=10)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f'--- Page {meta["page"]} (relevance: {round(1-dist,3)}) ---')
    print(doc)
    print()
