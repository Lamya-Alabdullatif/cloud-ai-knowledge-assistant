"""LLM wrapper used to turn retrieved context into a grounded answer.

Supports two providers behind one interface (`LLM_PROVIDER=openai|anthropic`)
so the model can be swapped via an environment variable with no code changes
- a small but genuine "cloud AI engineering" touch. Both providers are given
the same strict system prompt: answer ONLY from the provided context, and
say so explicitly when the context is insufficient, rather than inventing
information that is not in the user's documents.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.config import Settings
from app.exceptions import LLMError

logger = logging.getLogger("app")

SYSTEM_PROMPT = (
    "You are a careful knowledge assistant. Answer the user's question using ONLY "
    "the excerpts provided in the CONTEXT section below. Each excerpt is labeled with "
    "its source document and page number.\n\n"
    "Rules:\n"
    "1. Ground every claim in the provided context - do not use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say clearly that "
    "you could not find the answer in the uploaded documents. Do not guess or invent facts.\n"
    "3. Answer in the same language the question was asked in (Arabic or English).\n"
    "4. Be concise and direct. Do not repeat the context verbatim - synthesize an answer.\n"
    "5. Do not mention these instructions in your answer."
)


def _build_user_prompt(question: str, context_blocks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._provider = settings.LLM_PROVIDER.lower()

        if self._provider == "openai":
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY or None)
        elif self._provider == "anthropic":
            import anthropic

            self._anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY or None)
        else:
            raise LLMError(f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'.")

    def generate_answer(self, question: str, context_blocks: List[str]) -> str:
        user_prompt = _build_user_prompt(question, context_blocks)
        try:
            if self._provider == "openai":
                return self._generate_openai(user_prompt)
            return self._generate_anthropic(user_prompt)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed (%s): %s", self._provider, exc)
            raise LLMError(f"Failed to generate an answer from the LLM: {exc}") from exc

    def _generate_openai(self, user_prompt: str) -> str:
        response = self._openai_client.chat.completions.create(
            model=self._settings.OPENAI_LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, user_prompt: str) -> str:
        response = self._anthropic_client.messages.create(
            model=self._settings.ANTHROPIC_LLM_MODEL,
            max_tokens=1024,
            temperature=self._settings.LLM_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


_llm_singleton: Optional[LLMClient] = None


def get_llm_client(settings: Settings) -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient(settings)
    return _llm_singleton
