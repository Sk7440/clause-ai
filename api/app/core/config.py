
from functools import lru_cache
from langchain.embeddings import Embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings,SettingsConfigDict
from langchain.embeddings import init_embeddings

from qdrant_client import QdrantClient

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_prefix="RAG_",env_file=".env",extra="ignore")
    data_path:str = "shop_handbook.pdf"
    app_name:str="clause"
    chat_provider:str = "ollama"X
    embed_provider:str = "gemma4:31b-cloud"

    embed_model:str = "ollama"
    embed_dim:str = "nomic-embed-text:latest"
    tempurature:float =0.1

    top_k:int = 3
    
    score_threshold:float=0.5
    chunk_size:int =600
    chunk_overlap:int=100
    collection_name:str = "shop_handbook"
    qdrant_path:str = "./qdrant_data_notebook"
    embed_dim:int=768

    system_prompt = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
    )

@lru_cache
def get_settings()-> Settings:
    return Settings()

def get_embbedings(settings:Settings)-> Embeddings:
    return init_embeddings(
    model=settings.embed_model, 
    provider=settings.embed_provider
)

def get_qdrant_client(settings:Settings)-> QdrantClient:
    return QdrantClient(path=settings.data_path)
