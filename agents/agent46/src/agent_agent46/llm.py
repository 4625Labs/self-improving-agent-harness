"""LLM provider selection for Agent46."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.language_models import BaseChatModel

MODEL = "gpt-5.6-luna"


def build_llm(temperature: float = 0) -> BaseChatModel:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is missing; add it to .env or project secrets")
    return ChatOpenAI(
        model=MODEL,
        api_key=SecretStr(api_key),
        use_responses_api=True,
    )
