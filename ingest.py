import json
import chromadb

def build_vector_index():
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(name="jobs_index")
    except Exception:
        pass

    collection = client.create_collection(
        name="jobs_index",
        metadata={"hnsw:space": "cosine"}
    )

    with open("jobs_data.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)

    documents, metadatas, ids = [], [], []

    for job in jobs:
        semantic_text = (
            f"Role: {job['title']} at {job['company']}. "
            f"Location: {job['location']}. "
            f"Skills: {', '.join(job['skills'])}. "
            f"Description: {job['description']}"
        )
        documents.append(semantic_text)
        metadatas.append({
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "stipend_or_ctc": job["stipend_or_ctc"],
            "skills": ", ".join(job["skills"])
        })
        ids.append(job["id"])

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Successfully indexed {len(ids)} jobs into ChromaDB!")

if __name__ == "__main__":
    build_vector_index()