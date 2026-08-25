"""Embedding generation.

Uses OpenAI's embedding models (multilingual, good Arabic + English support)
via the official `openai` SDK. Isolated behind this small class so the rest
of the app never talks to the OpenAI SDK directly.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from openai import APIError, OpenAI

from app.config import Settings
from app.exceptions import LLMError

logger = logging.getLogger("app")


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = settings.EMBEDDING_MODEL
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY or None)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except APIError as exc:
            logger.error("Embedding request failed: %s", exc)
            raise LLMError(f"Failed to generate embeddings: {exc}") from exc
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


_embedding_singleton: Optional[EmbeddingClient] = None


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = EmbeddingClient(settings)
    return _embedding_singleton
