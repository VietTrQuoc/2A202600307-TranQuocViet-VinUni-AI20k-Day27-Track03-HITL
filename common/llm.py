"""LLM factory for OpenAI direct usage, with OpenRouter fallback."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        kwargs = {
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "api_key": openai_key,
            "temperature": temperature,
        }
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return ChatOpenAI(**kwargs)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key,
            temperature=temperature,
        )

    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to .env, or set OPENROUTER_API_KEY "
        "to use the OpenRouter fallback."
    )
