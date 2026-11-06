from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from core.config import get_settings,get_embeddings,get_qdrant_client


class Source(BaseModel):
    source:str
    page:int|None=None



class Answer():
    answer:str
    sources:list[Source]

class RAGService():
    def __init__(self)->None:
        self.settings=get_settings()
        self.client=get_qdrant_client(self.settings)
        if not self.client:
            raise RuntimeError(f"Collection{self.settings.collection_name} not found!")
        self.store=QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings(self.settings)

        )
        self.llm=init_chat_model(
            model=self.settings.chat_models,
            model_provider=self.settings.chat_provider,
            embeddings=get_embeddings(self.settings)
        )

    def ask(self,question:str)->None:
        pass
    def close(self)->None:
        pass

def main():
    rag_service=RAGService()
    try:
        query=input("you:").strip()
        result= rag_service.ask(query)
    finally:
        rag_service.close()