"""
llm/gemini_client.py — Google Gemini API client for answer generation.

Clean provider boundary: all Gemini-specific calls live here.
Uses prompts/templates.py for prompt construction.
"""

import os
import time
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts.templates import build_system_instruction, build_user_prompt

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("llm", {})
    except Exception:
        return {}


def generate(query: str, context_chunks: list[dict]) -> dict:
    """
    Generate a grounded, source-cited answer using Gemini.

    Args:
        query: User question string.
        context_chunks: Top-N chunks from retrieval/reranker.

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file."
        )

    if not context_chunks:
        return {"answer": "İlgili bilgi bulunamadı.", "sources": []}

    cfg = _load_config()
    model_name = cfg.get("model", "gemini-flash-latest")
    max_output_tokens = cfg.get("max_output_tokens", 1024)
    temperature = cfg.get("temperature", 0.2)
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_delay_seconds", 5)

    # Build sources list for the response
    sources = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_name = chunk["metadata"].get("source", "unknown")
        sources.append({
            "number": i,
            "id": chunk["id"],
            "source": source_name,
            "text_preview": chunk["text"][:200],
        })

    system_instruction = build_system_instruction()
    user_prompt = build_user_prompt(query, context_chunks)

    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[LLM] Sending generation request to Gemini ({model_name})...")

    answer = ""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            answer = response.text.strip()
            print(f"[LLM] Answer generated ({len(answer)} chars).")
            break
        except Exception as e:
            print(f"[LLM] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                answer = (
                    "[FALLBACK] Gemini API is unavailable. "
                    "Context was retrieved successfully. [1]"
                )
                print("[LLM] Using fallback answer.")

    return {"answer": answer, "sources": sources}
