from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from core.config import get_settings,get_embeddings,get_qdrant_client

NO_Answer="I dont know. the handbook dosen't cover that "
class Source(BaseModel):
    source:str
    page:int|None=None



class Answer():
    answer:str
    sources:list[Source]
    score:float

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
        question=question.strip()
        if not question:
            raise ValueError("Question must be not Empty")
            results=self.store,similarty_search_with_score(
                question,
                k=self.settings.top_k
            )

        if not results:
            return Answer(answer=NO_Answer,sources=[])
        context="\n\n".join(doc.page_content for doc,_ in results)
        messages=[{"role":"system","content":self.settings.system_prompt},{"role":"user","content":f"Context:\n{context}\n\nQuestion:{question}"}]
        return Answer(
            answer=self.llm.invoke(messages).text,
            sources=[Source(
            pages=doc.metadata.get("pages"),
            score=round(score,4)
            

            )]
        )
    def close(self)->None:
        pass

