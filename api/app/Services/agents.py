from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from app.core.config import get_settings
from app.prompts.loader import get_prompt
from app.llm.models import get_chat_model
from app.rag.vector_store import get_vector_store

class ClauseAgent:
    def __init__(self) -> None:
        self.rag = get_settings()
        self.client = get_vector_store(self.settings)

        @tool(response_format="content_and_artifact")
        def search_handbook(question: str):
            """Search the Clause shop handbook: refunds, delivery times and fees,
            payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
            Use a short query like "refund window electronics"."""
            results = self.client.store.similarity_search_with_score(
                query=question,
                k=self.client.settings.top_k,
            )
            if not results:
                return "Nothing found.", []

            texts = []
            sources = []
            for doc, score in results:
                texts.append(doc.page_content)
                sources.append(
                    {
                        "sources": doc.metadata.get("sources"),
                        "page": doc.metadata.get("page"),
                        "score": round(score, 4),
                    }
                )

            combined_text = "\n\n".join(texts)
            return combined_text, sources

        self.agent = create_agent(
            model=get_chat_model(self.rag.settings),
            tools=[search_handbook],
            system_prompt=get_prompt(
                name="support/system",
                version=self.client.settings.system_prompt_version
                ),
        )

    def ask(self, question: str) -> str:
        result = self.client.agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]
        return messages[-1].text

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.store.client.close()