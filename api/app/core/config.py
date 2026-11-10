
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain.embeddings import init_embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore"
    )

    # Application settings
    data_path: str = "shop_handbook.pdf"
    app_name: str = "clause"

    # AI provider settings
    chat_provider: str = "ollama"
    embed_provider: str = "ollama"

    # Model settings
    chat_model: str = "gemma4:31b-cloud"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # RAG settings
    temperature: float = 0.1
    top_k: int = 3
    score_threshold: float = 0.5

    # Document settings
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Vector database settings
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data_notebook"

    # System prompt
    system_prompt: str = (
        "You are a helpful Clause Shop assistant. "
        "Answer only from the provided context. "
        "If the answer is not in the context, say you don't know."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider
    )


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(path=settings.qdrant_path)
