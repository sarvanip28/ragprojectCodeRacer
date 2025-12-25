# app/main.py
import os
import time
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio

load_dotenv()

# package-relative imports (important when running `uvicorn app.main:app`)
from .llm_client import call_model
from .rag import retrieve_similar

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "bigcode/starcoder")
PREFER = os.getenv("PREFER_MODEL", "hf")  # 'hf' or other preference keys your llm_client supports

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("coderacer")

app = FastAPI(title="CodeRacer - RAG + HF API")


class ExplainRequest(BaseModel):
    code: str = Field(..., description="Full source code text")
    language: str = Field(..., description="Programming language (e.g., python, javascript)")
    line_index: int = Field(..., ge=0, description="Zero-based line index to center the snippet on")
    session_id: Optional[str] = None
    action: Optional[str] = Field("explain", description="'explain' | 'fix' | 'summarize'")


class ExplainResponse(BaseModel):
    explanation: dict
    start_line: int
    end_line: int
    took_ms: int


def get_snippet_window(code: str, line_index: int, window: int = 6) -> Tuple[str, int, int]:
    """
    Return a snippet centered around `line_index` with +/- window lines.
    Ensures indices are within bounds.
    """
    lines = code.splitlines()
    if not lines:
        return "", 0, 0
    # clamp line_index
    idx = max(0, min(line_index, len(lines) - 1))
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    snippet = "\n".join(lines[start:end])
    return snippet, start, end


def make_prompt_explain(snippet: str, language: str, context_texts: Optional[List[str]]):
    """
    Build the prompt for the model. Keep the context truncated so prompt doesn't explode.
    """
    # join contexts with a readable separator, limit each context length
    if context_texts:
        ctx = "\n\n---\n".join((t[:1500] for t in context_texts))
    else:
        ctx = ""

    prompt = f"""You are an expert {language} programmer and teacher.

Task: Provide a JSON object with these keys:
1) "lines": an array where each item is an object: {{ "line_no": <int>, "text": "<line text>", "explain": "<short explanation>", "issue": "<issue or null>" }}
2) "summary": a short (2-3 sentence) summary of the selected snippet.
3) "fixes": suggested fixes (code snippets) if issues exist.

Selected snippet:
{snippet}

Context / references:
{ctx}

Return ONLY valid JSON (no extra commentary).
"""
    return prompt


@app.get("/health")
def health():
    try:
        # quick check whether RAG index works (non-invasive)
        available = bool(retrieve_similar("test", k=1))
    except Exception as e:
        logger.warning("RAG health check failed: %s", e)
        available = False
    return {"status": "ok", "rag_available": available}


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    t0 = time.time()

    snippet, start, end = get_snippet_window(req.code, req.line_index, window=6)
    if snippet == "":
        raise HTTPException(status_code=400, detail="Provided code is empty or invalid.")

    # retrieve RAG context (safe-guard exceptions)
    try:
        context = retrieve_similar(snippet, k=5) or []
    except Exception as e:
        logger.exception("RAG retrieval failed, continuing without context")
        context = []

    prompt = make_prompt_explain(snippet, req.language, context)

    model = DEFAULT_HF_MODEL
    prefer = PREFER

    # If call_model is a blocking function, run it in a thread to avoid blocking the event loop.
    try:
        out = await asyncio.to_thread(call_model, prompt, prefer, model)
    except TypeError:
        # fallback if call_model expects keyword args or different signature
        try:
            out = await asyncio.to_thread(call_model, prompt, prefer=prefer, hf_model=model)
        except Exception as e:
            logger.exception("call_model failed")
            raise HTTPException(status_code=500, detail=f"Model call failed: {e}")
    except Exception as e:
        logger.exception("call_model raised exception")
        raise HTTPException(status_code=500, detail=f"Model call failed: {e}")

    took_ms = int((time.time() - t0) * 1000)
    # Attempt to parse model output as JSON if call_model returns str; otherwise pass through.
    explanation = out
    return {"explanation": explanation, "start_line": start, "end_line": end, "took_ms": took_ms}
