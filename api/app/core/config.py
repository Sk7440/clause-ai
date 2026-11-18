from functools import lru_cache
from pathlib import Path

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient


# api/.env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=ENV_FILE, extra="ignore"
    )

    # App
    app_name: str = "Clause"
    data_path: str = "files/shop_handbook.pdf"

    # Chat model
    chat_provider: str = "ollama"
    chat_model: str = "gemma3:1b"
    chat_temperature: float = 0.1
    chat_api_key: str = ""

    # Embeddings
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 3
    score_threshold: float = 0.5

    # Qdrant
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./vector_stores"
    qdrant_url: str | None = None

    # Prompts
    # system_prompt: str = (
    #     "You are a helpful Clause assistant. "
    #     "Answer only from the provided context. "
    #     "If the answer is not in the context, say you don't know."
    # )

    system_prompt: str = (
        "You are Clause, the customer support assistant for Clause shop.\n"
        "Your only source of truth is the Shop handbook, which you read with the search_handbook tool.\n"
        "\n"
        "# When to search\n"
        "- Questions about refunds, delivery, payments, warranty, exchanges, order tracking,\n"
        "  loyalty points, data privacy or support hours: call search_handbook before answering.\n"
        "- A question with two parts: search once for each part, then answer both.\n"
        "- Greetings, thanks and small talk: reply briefly and warmly. Do not search.\n"
        "\n"
        "# How to answer\n"
        "- Use only facts from the search results. Never guess, assume or use outside knowledge.\n"
        "- Copy numbers, prices, time limits, emails and phone numbers exactly as written.\n"
        "- Keep it to 1-2 short, friendly sentences in plain language.\n"
        "- Speak as Shop (\"we\", \"our\"). Never mention the handbook, search results or tools.\n"
        "- Never share anything marked \"Internal\" or \"staff only\".\n"
        "\n"
        "# When the handbook does not cover it\n"
        "If the results do not clearly answer the question, do not guess. Reply exactly:\n"
        "\"I don't know. Our support team can help on live chat or at help@shop.com.\""
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider,
    )

def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(path=settings.qdrant_path)