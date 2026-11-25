import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from agent import chat_with_agent, ChatRequest

load_dotenv()
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TEST_CASES = [
    {
        "query": "Looking for an AI agent internship in Gurugram using Python and tool calling.",
        "expected_job_id": "job_01"
    },
    {
        "query": "Remote RAG engineer role working with ChromaDB and LLM evaluations.",
        "expected_job_id": "job_03"
    },
    {
        "query": "Backend Python developer with FastAPI and PostgreSQL in Bengaluru.",
        "expected_job_id": "job_02"
    },
    {
        "query": "Machine learning recommendation systems engineer in Gurugram.",
        "expected_job_id": "job_05"
    }
]


class JudgeScore(BaseModel):
    groundedness_score: int = Field(description="Score 1-5: Are explanations factual and grounded in job skills?")
    reason: str = Field(description="Brief justification for the score")


def call_with_retry(fn, retries=6, delay=10):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (attempt + 1)
            print(f"  [Model busy (503), waiting {wait}s before retry {attempt + 1}/{retries - 1}...]")
            time.sleep(wait)


def run_evaluations():
    hits_at_2 = 0
    total_judge_score = 0

    print("Running RoleMatch-Agent Evaluation Suite...\n" + "-" * 50)

    for idx, case in enumerate(TEST_CASES, start=1):
        res = call_with_retry(
            lambda: chat_with_agent(ChatRequest(session_id=f"eval_{idx}", message=case["query"]))
        )
        top_ids = [rec.job_id for rec in res.recommendations[:2]]
        is_hit = case["expected_job_id"] in top_ids
        if is_hit:
            hits_at_2 += 1

        judge_prompt = (
            f"User Query: {case['query']}\n"
            f"Agent Recommendations: {[r.model_dump() for r in res.recommendations[:2]]}\n"
            "Evaluate if the recommended jobs and 'why_it_fits' explanations accurately match the user query without hallucinating."
        )
        judge_res = call_with_retry(
            lambda: ai_client.models.generate_content(
                model="gemini-4-hydrogen",
                contents=judge_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeScore,
                    temperature=0.0,
                ),
            )
        )
        score_obj = JudgeScore.model_validate_json(judge_res.text)
        total_judge_score += score_obj.groundedness_score

        print(f"Test {idx}: Hit@2 = {is_hit} | Top IDs = {top_ids} | Judge Score = {score_obj.groundedness_score}/5")
        time.sleep(2)

    precision_at_2 = (hits_at_2 / len(TEST_CASES)) * 100
    avg_groundedness = total_judge_score / len(TEST_CASES)
    print("-" * 50)
    print(f"Final Precision@2: {precision_at_2:.1f}% | Avg Groundedness Score: {avg_groundedness:.2f}/5.0")


if __name__ == "__main__":
    run_evaluations()