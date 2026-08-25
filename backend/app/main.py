"""FastAPI application entrypoint.

Exposes the REST API described in the README:

  POST   /upload              upload a PDF and kick off processing
  GET    /documents           list all tracked documents + status
  GET    /documents/{id}      get one document's metadata/status
  DELETE /documents/{id}      delete a document (S3 + Qdrant + registry)
  POST   /ask                 ask a question, get a grounded answer + sources
  GET    /health               liveness/readiness probe (also used by CloudWatch)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.dependencies import get_document_service, get_rag_pipeline
from app.document_service import DocumentService
from app.exceptions import InvalidQuestionError, register_exception_handlers
from app.logging_config import configure_logging
from app.rag_pipeline import RAGPipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    HealthResponse,
    UploadResponse,
)

settings = get_settings()
logger = configure_logging(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting %s (env=%s, llm_provider=%s)",
        settings.APP_NAME,
        settings.APP_ENV,
        settings.LLM_PROVIDER,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Cloud-based Retrieval-Augmented Generation assistant: upload PDFs, "
    "ask questions, get grounded answers with sources.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Simple liveness probe - also useful as a CloudWatch/uptime check target."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        version=__version__,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"], status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    """Upload a PDF: validated, stored in S3, then processed synchronously
    (extracted, chunked, embedded, and indexed in Qdrant) before returning."""
    file_bytes = await file.read()
    record = service.upload_and_process(filename=file.filename, file_bytes=file_bytes)
    return UploadResponse(
        document_id=record.document_id,
        document_name=record.document_name,
        status=record.status,
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@app.get("/documents/{document_id}", response_model=DocumentMetadata, tags=["Documents"])
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    return service.get_document(document_id)


@app.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DeleteResponse:
    service.delete_document(document_id)
    return DeleteResponse(document_id=document_id)


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> AskResponse:
    if not request.question or not request.question.strip():
        raise InvalidQuestionError("The question must not be empty.")

    return pipeline.ask(
        question=request.question,
        document_ids=request.document_ids,
        top_k=request.top_k,
    )
