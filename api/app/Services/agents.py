from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from app.services.chat import RAGService


class ClauseAgent:
    def __init__(self) -> None:
        self.rag = RAGService()

        @tool(response_format="content_and_artifact")
        def search_handbook(question: str):
            """Search the Clause shop handbook: refunds, delivery times and fees,
            payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
            Use a short query like "refund window electronics"."""
            results = self.rag.store.similarity_search_with_score(
                query=question,
                k=self.rag.settings.top_k,
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
            model=self.rag.llm,
            tools=[search_handbook],
            system_prompt=self.rag.settings.system_prompt,
        )

    def ask(self, question: str) -> str:
        result = self.agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]
        return messages[-1].text

    def close(self) -> None:
        if hasattr(self.rag, "close"):
            self.rag.close()