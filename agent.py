import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini client & ChromaDB vector collection
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="jobs_index")

app = FastAPI(title="RoleMatch-Agent API", version="1.0")

# In-memory session store for multi-turn conversation context
SESSION_MEMORY: Dict[str, List[str]] = {}


# --- 1. Pydantic Schemas for Structured Outputs ---
class MatchedJob(BaseModel):
    job_id: str = Field(description="Unique ID of the job from the search tool")
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    stipend_or_ctc: str = Field(description="Compensation details")
    fit_score: int = Field(description="Relevance score from 0 to 100 based on user profile")
    why_it_fits: str = Field(description="Concise, friend-like explanation of why this role matches the user")


class AgentResponse(BaseModel):
    session_id: str
    agent_summary: str = Field(description="Helpful conversational summary addressing the user's request")
    recommendations: List[MatchedJob] = Field(description="Ranked list of best matching jobs")


class ChatRequest(BaseModel):
    session_id: str = "chhaya_session_1"
    message: str


# --- 2. Tool Definition for Semantic Vector Search ---
def search_jobs_semantic(query: str, top_k: int = 4) -> list[dict]:
    """Searches the vector index of open jobs using semantic similarity.

    Args:
        query: Natural language description of desired role, skills, or location.
        top_k: Number of top matching jobs to retrieve (default 4).
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    matches = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        matches.append({
            "job_id": results["ids"][0][i],
            "title": meta["title"],
            "company": meta["company"],
            "location": meta["location"],
            "stipend_or_ctc": meta["stipend_or_ctc"],
            "skills": meta["skills"],
            "similarity_score": round(1 - distance, 3)
        })
    return matches


# --- 3. FastAPI Endpoint Combining Tool Use + Memory + Structured Output ---
@app.post("/chat", response_model=AgentResponse)
def chat_with_agent(req: ChatRequest):
    history = SESSION_MEMORY.setdefault(req.session_id, [])
    history.append(f"User: {req.message}")
    conversation_context = "\n".join(history[-6:])  # Keep last 6 turns of context

    # Step A: Agent uses tool calling to retrieve relevant jobs from ChromaDB
    tool_prompt = (
        "You are RoleMatch-Agent, a career matching assistant. "
        "Use the `search_jobs_semantic` tool to find relevant roles based on the conversation below.\n\n"
        f"Conversation History:\n{conversation_context}"
    )

    tool_response = ai_client.models.generate_content(
        model="gemini-4-hydrogen",
        contents=tool_prompt,
        config=types.GenerateContentConfig(
            tools=[search_jobs_semantic],
            temperature=0.2,
        ),
    )

    # Step B: Rank & format final response using strict Pydantic Structured Outputs
    ranking_prompt = (
        "Based on the user's conversation and the retrieved job candidates below, "
        "rank the top matches and explain clearly why each fits like a knowledgeable friend.\n\n"
        f"Conversation:\n{conversation_context}\n\n"
        f"Retrieved Job Candidates & Tool Notes:\n{tool_response.text}"
    )

    structured_response = ai_client.models.generate_content(
        model="gemini-4-hydrogen",
        contents=ranking_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgentResponse,
            temperature=0.2,
        ),
    )

    parsed = AgentResponse.model_validate_json(structured_response.text)
    parsed.session_id = req.session_id
    history.append(f"Agent: {parsed.agent_summary}")
    return parsed


@app.get("/")
def health_check():
    return {"status": "running", "indexed_jobs": collection.count()}