# app/llm_client.py
"""Lightweight LLM client with huggingface + openai support, retries and async helper."""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, Any, Callable

from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("coderacer.llm_client")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PREFERRED = (os.getenv("PREFERRED_LLM") or "").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# optional import of openai
try:
    import openai  # type: ignore
except Exception:
    openai = None  # pragma: no cover

# Small configurable retry policy
_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))
_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY_SECONDS", "0.5"))


def _retry(fn: Callable[..., Any], attempts: int = _RETRY_ATTEMPTS, delay: float = _RETRY_DELAY, *args, **kwargs):
    """Simple retry loop with fixed delay. Returns result or raises last exception."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            LOG.warning("Attempt %d/%d failed for %s: %s", i + 1, attempts, getattr(fn, "__name__", str(fn)), e)
            if i + 1 < attempts:
                time.sleep(delay)
    raise last_exc


def call_hf_inference(prompt: str, model: str = "bigcode/starcoder", max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call the Hugging Face Inference API (sync).

    Returns the generated text as str. Raises RuntimeError on missing config or underlying errors.
    """
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN not set in environment")

    # import lazily so module is optional
    try:
        from huggingface_hub import InferenceApi
    except Exception as e:  # pragma: no cover
        raise RuntimeError("huggingface_hub is not installed") from e

    def _call():
        api = InferenceApi(repo_id=model, token=HF_TOKEN)
        params = {"max_new_tokens": max_new_tokens, "temperature": float(temperature)}
        resp = api(inputs=prompt, params=params)

        # Typical responses: dict with 'generated_text', just a str, or a list
        if isinstance(resp, dict):
            # huggingface sometimes returns {"error": "..."} or {"generated_text": "..."}
            if "generated_text" in resp:
                return str(resp["generated_text"])
            # some models return {"error": ...}
            if "error" in resp:
                raise RuntimeError(f"HF inference error: {resp['error']}")
            # fallback stringification
            return str(resp)
        if isinstance(resp, list):
            # list of outputs - join or take first sensible item
            first = resp[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"])
            return str(first)
        if isinstance(resp, str):
            return resp
        return str(resp)

    return _retry(_call)


def call_openai_chat(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Call OpenAI ChatCompletion (sync). Raises RuntimeError if openai not configured.
    """
    if not OPENAI_API_KEY or not openai:
        raise RuntimeError("OpenAI not configured or openai library not installed")
    openai.api_key = OPENAI_API_KEY

    def _call():
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # defensive extraction
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenAI response shape: {e}") from e

    return _retry(_call)


def call_model(prompt: str, prefer: Optional[str] = None, hf_model: str = "bigcode/starcoder") -> str:
    """
    Unified synchronous call. Tries preferred backend, then fallbacks.

    prefer: "hf" or "openai" (case-insensitive). If None, uses PREFERRED env var.
    """
    p = (prefer or PREFERRED or "").lower()
    errors = []

    # ensure prompt is string
    if prompt is None:
        raise ValueError("prompt must be a string")
    prompt = str(prompt)

    # helper to attempt backends cleanly
    def _try_hf():
        try:
            return call_hf_inference(prompt, model=hf_model)
        except Exception as e:
            errors.append(("hf", str(e)))
            LOG.debug("HF attempt error: %s", e)
            raise

    def _try_openai():
        try:
            return call_openai_chat(prompt)
        except Exception as e:
            errors.append(("openai", str(e)))
            LOG.debug("OpenAI attempt error: %s", e)
            raise

    # prefer explicit
    if p == "hf":
        try:
            return _try_hf()
        except Exception:
            pass
        if OPENAI_API_KEY and openai:
            try:
                return _try_openai()
            except Exception:
                pass
    elif p == "openai":
        if OPENAI_API_KEY and openai:
            try:
                return _try_openai()
            except Exception:
                pass
        if HF_TOKEN:
            try:
                return _try_hf()
            except Exception:
                pass
    else:
        # no explicit preference: try HF first (if token), then OpenAI
        if HF_TOKEN:
            try:
                return _try_hf()
            except Exception:
                pass
        if OPENAI_API_KEY and openai:
            try:
                return _try_openai()
            except Exception:
                pass

    # nothing worked
    raise RuntimeError(f"No LLM available. Errors: {errors}")


# convenience async wrapper
async def call_model_async(prompt: str, prefer: Optional[str] = None, hf_model: str = "bigcode/starcoder") -> str:
    """
    Async wrapper using threadpool for blocking call_model.
    """
    import asyncio

    return await asyncio.to_thread(call_model, prompt, prefer, hf_model)
