
from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import (
    get_settings,
    get_embeddings,
    get_qdrant_client
)

NO_ANSWER = "I don't know. The handbook doesn't cover that."


class Source(BaseModel):
    source: str
    page: int | None = None


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    score: float = 0.0


class RAGService:

    def __init__(self) -> None:

        self.settings = get_settings()
        self.client = get_qdrant_client(self.settings)

        # Check collection existence
        if not self.client.collection_exists(
            self.settings.collection_name
        ):
            raise RuntimeError(
                f"Collection {self.settings.collection_name} not found!"
            )

        # Initialize vector store
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings(self.settings)
        )

        # Initialize LLM
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.temperature
        )

    def ask(self, question: str) -> Answer:

        # 1. Validate question
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty")

        # 2. Search relevant documents
        results = self.store.similarity_search_with_score(
            question,
            k=self.settings.top_k
        )

        # 3. Check results
        if not results:
            return Answer(
                answer=NO_ANSWER,
                sources=[]
            )

        # 4. Prepare context
        context = "\n\n".join(
            doc.page_content
            for doc, score in results
        )

        # 5. Prepare messages
        messages = [
            {
                "role": "system",
                "content": self.settings.system_prompt
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]

        # 6. Generate response
        response = self.llm.invoke(messages)

        # 7. Extract sources
        sources = [
            Source(
                source=str(doc.metadata.get("source", "Unknown")),
                page=doc.metadata.get("page")
            )
            for doc, score in results
        ]

        # 8. Return answer
        return Answer(
            answer=response.content,
            sources=sources,
            score=round(
                sum(score for doc, score in results) / len(results),
                4
            )
        )

    def close(self) -> None:
        pass
