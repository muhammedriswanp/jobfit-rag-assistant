import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.EphemeralClient()

resume_collection = client.get_or_create_collection("resume")
jd_collection = client.get_or_create_collection("job_description")

def add_to_collection(collection, chunks: list[str], doc_type: str):
    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{doc_type}_{i}" for i in range(len(chunks))]
    metadatas = [{"type": doc_type} for _ in chunks]

    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings
    )

def retrieve(query: str, collection, k: int=5) -> list[str]:
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    ) 

    return results['documents'][0]

