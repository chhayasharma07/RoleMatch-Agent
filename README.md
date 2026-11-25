
# RoleMatch-Agent — Conversational Job Retrieval & Ranking Engine

An LLM-powered job matching and semantic search backend built with **Python, FastAPI, Google Gemini (`google-genai`), ChromaDB, Pydantic, and Docker**.

## Key Features
- **Semantic Vector Search:** Indexes job descriptions, skills, and locations into **ChromaDB** (`all-MiniLM-L6-v2` embeddings with cosine similarity).
- **Agentic Tool Use & Memory:** Multi-turn conversational agent that remembers user preferences across turns and invokes `search_jobs_semantic` dynamically.
- **Structured Outputs:** Enforces **Pydantic** JSON schemas (`AgentResponse`) to return ranked job matches, fit scores (0–100), and plain-English fit explanations.
- **Automated LLM Evaluations:** Includes `eval.py` measuring retrieval `Precision@2` and `LLM-as-a-Judge` groundedness scores.
- **Dockerized Deployment:** Production-ready `Dockerfile` that builds the vector index and serves the FastAPI app on port `8000`.

## Quickstart
```bash
pip install -r requirements.txt
python ingest.py
uvicorn agent:app --reload