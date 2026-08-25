"""
Centralized application configuration.

All configuration is read from environment variables (optionally loaded from a
local `.env` file via python-dotenv / pydantic-settings). Nothing sensitive is
hardcoded here - see `.env.example` at the project root for the full list of
variables this application expects.
"""
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    APP_ENV: str = Field(default="development", description="development|production")
    APP_NAME: str = Field(default="Cloud AI Knowledge Assistant")
    LOG_LEVEL: str = Field(default="INFO")
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    CORS_ORIGINS: str = Field(default="*", description="Comma-separated list of allowed origins")

    # ------------------------------------------------------------------ #
    # AWS
    # ------------------------------------------------------------------ #
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")
    AWS_REGION: str = Field(default="us-east-1")
    S3_BUCKET_NAME: str = Field(default="")

    # ------------------------------------------------------------------ #
    # CloudWatch monitoring (optional but recommended)
    # ------------------------------------------------------------------ #
    CLOUDWATCH_ENABLED: bool = Field(default=False)
    CLOUDWATCH_LOG_GROUP: str = Field(default="/cloud-ai-knowledge-assistant/app")
    CLOUDWATCH_LOG_STREAM: str = Field(default="backend")

    # ------------------------------------------------------------------ #
    # Qdrant Cloud
    # ------------------------------------------------------------------ #
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION_NAME: str = Field(default="knowledge_assistant_chunks")

    # ------------------------------------------------------------------ #
    # Embeddings + LLM
    # ------------------------------------------------------------------ #
    OPENAI_API_KEY: str = Field(default="")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = Field(default=1536)

    LLM_PROVIDER: str = Field(default="openai", description="openai|anthropic")
    OPENAI_LLM_MODEL: str = Field(default="gpt-4o-mini")
    ANTHROPIC_API_KEY: str = Field(default="")
    ANTHROPIC_LLM_MODEL: str = Field(default="claude-3-5-haiku-latest")
    LLM_TEMPERATURE: float = Field(default=0.1)

    # ------------------------------------------------------------------ #
    # RAG / retrieval tuning
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = Field(default=1000, description="Characters per chunk")
    CHUNK_OVERLAP: int = Field(default=150, description="Character overlap between chunks")
    TOP_K_RESULTS: int = Field(default=5, description="Chunks retrieved per question")
    SIMILARITY_SCORE_THRESHOLD: float = Field(
        default=0.25, description="Minimum cosine similarity to trust a retrieved chunk"
    )

    # ------------------------------------------------------------------ #
    # Upload validation
    # ------------------------------------------------------------------ #
    ALLOWED_EXTENSIONS: str = Field(default=".pdf")
    MAX_FILE_SIZE_MB: int = Field(default=20)

    # ------------------------------------------------------------------ #
    # Local metadata registry (document status tracking)
    # ------------------------------------------------------------------ #
    DOCUMENTS_REGISTRY_PATH: str = Field(default="data/documents_registry.json")

    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def _normalize_extensions(cls, value: str) -> str:
        return value.lower()

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
