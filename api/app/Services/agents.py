from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from app.core.config import get_settings
from app.prompts.loader import get_prompt
from app.llm.models import get_chat_model
from app.rag.vector_store import get_vector_store
from app.tools.knowledge.handbook import Source, file_search


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class ClauseAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = get_vector_store(self.settings)

        self.agent = create_agent(
            model=get_chat_model(self.settings),
            tools=[file_search(self.store, self.settings.top_k)],
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