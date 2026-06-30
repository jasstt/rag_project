"""
prompts/ — Separates prompt templates from model-provider logic.
"""
from .templates import build_system_instruction, build_user_prompt, build_rerank_prompt

__all__ = ["build_system_instruction", "build_user_prompt", "build_rerank_prompt"]
