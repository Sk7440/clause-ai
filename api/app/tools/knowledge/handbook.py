from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from app.schemas.retrival import Source

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float


def file_search(store: QdrantVectorStore, top_k: int):
    @tool(response_format="content_and_artifact")
    def search_handbook(question: str):
        """Search the Clause shop handbook: refunds, delivery times and fees,
        payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
        Use a short query like "refund window electronics"."""
        results = store.similarity_search_with_score(
            query=question,
            k=top_k,
        )
        if not results:
            return "Nothing found.", []

        # texts = []
        # sources = []
        # for doc, score in results:
        #     texts.append(doc.page_content)
        #     sources.append(
        #         Source(
        #             source=doc.metadata.get("source", "unknown"),
        #             page=doc.metadata.get("page"),
        #             score=round(score, 4),
        #         )
        #     )

        combined_text = "\n\n".join(doc.page_content for doc, _ in results)
        sources = [
            Source(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                score=round(score, 4),
            )
            for doc, score in results
        ]
        return combined_text, sources

    return search_handbook