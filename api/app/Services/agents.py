from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel
from app.core.config import get_settings
from app.prompts.loader import get_prompt
from app.llm.models import get_chat_model
from app.rag.vector_store import get_vector_store
from app.tools.knowledge.handbook import file_search

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class ClauseAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = get_vector_store(self.settings)

        # @tool(response_format="content_and_artifact")
        # def search_handbook(question: str)-> tuple[str, list[Source]]:
        #     """Search the Clause shop handbook: refunds, delivery times and fees,
        #     payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
        #     Use a short query like "refund window electronics"."""
        #     results = self.store.similarity_search_with_score(
        #         query=question,
        #         k=self.settings.top_k,
        #     )
        #     if not results:
        #         return "Nothing found.", []

        #     texts = []
        #     sources = []
        #     for doc, score in results:
        #         texts.append(doc.page_content)
        #         sources.append(
        #             Source(
        #                 source=doc.metadata.get("source", "unknown"),
        #                 page=doc.metadata.get("page"),
        #                 score=round(score, 4),
        #             )
        #         )

        #     combined_text = "\n\n".join(texts)
        #     return combined_text, sources

        self.agent = create_agent(
            model=get_chat_model(self.settings),
            tools=[search_handbook],
            system_prompt=get_prompt(
                name="support/system",
                version=self.settings.system_prompt_version,
            ),
        )

    def ask(self, question: str) -> Answer:
        result = self.agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]

        sources: list[Source] = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.artifact, list):
                sources.extend(msg.artifact)

        return Answer(answer=messages[-1].text, sources=sources)

    def close(self) -> None:
        if hasattr(self.store, "client") and hasattr(self.store.client, "close"):
            self.store.client.close()