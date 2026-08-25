"""The actual Retrieval-Augmented Generation pipeline.

Question -> query embedding -> Qdrant similarity search -> relevant chunks
-> context -> LLM -> grounded answer + sources.

This is a real retrieval pipeline, not "stuff the whole PDF into the
prompt": only the top-k most similar chunks (above a similarity threshold)
are ever sent to the LLM, and every answer is accompanied by the exact
document/page/chunk it was grounded in.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.config import Settings
from app.embeddings import EmbeddingClient
from app.llm_client import LLMClient
from app.schemas import AskResponse, SourceChunk
from app.vector_store import VectorStore

logger = logging.getLogger("app")

NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the uploaded documents to answer this question. "
    "Please try rephrasing your question, or upload a document that covers this topic.\n\n"
    "لم أجد معلومات كافية في المستندات المرفوعة للإجابة على هذا السؤال. "
    "حاول إعادة صياغة السؤال أو رفع مستند يغطي هذا الموضوع."
)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        llm_client: LLMClient,
    ):
        self._settings = settings
        self._embeddings = embedding_client
        self._vector_store = vector_store
        self._llm = llm_client

    def ask(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> AskResponse:
        top_k = top_k or self._settings.TOP_K_RESULTS

        query_vector = self._embeddings.embed_query(question)

        matches = self._vector_store.search(
            query_vector=query_vector, top_k=top_k, document_ids=document_ids
        )

        relevant = [m for m in matches if m["score"] >= self._settings.SIMILARITY_SCORE_THRESHOLD]

        if not relevant:
            logger.info(
                "No chunks passed the similarity threshold (%.2f) for question: %r",
                self._settings.SIMILARITY_SCORE_THRESHOLD,
                question,
            )
            return AskResponse(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                grounded=False,
            )

        context_blocks = [
            f"[Source: {m['document_name']}, page {m['page_number']}]\n{m['text']}"
            for m in relevant
        ]

        answer = self._llm.generate_answer(question=question, context_blocks=context_blocks)

        sources = [
            SourceChunk(
                document_id=m["document_id"],
                document_name=m["document_name"],
                page_number=m["page_number"],
                chunk_id=m["chunk_id"],
                score=round(m["score"], 4),
                text_snippet=(m["text"][:280] + "…") if len(m["text"]) > 280 else m["text"],
            )
            for m in relevant
        ]

        return AskResponse(question=question, answer=answer, sources=sources, grounded=True)
