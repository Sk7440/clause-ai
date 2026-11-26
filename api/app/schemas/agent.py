from typing import Annotated, Any

from pydantic import Field, StringConstraints, computed_field

from app.schemas.base import Schema
from app.schemas.retrieval import Source


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(Schema):
    """Body of POST /ask."""

    question: Question
    thread_id: str | None = None   # send it back to continue a conversation


class ToolCall(Schema):
    """A tool the LLM decided to call, with the arguments it chose."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentReply(Schema):
    """What ClauseAgent.ask() returns (and what POST /ask responds with)."""

    answer: str
    thread_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    @computed_field
    @property
    def searches(self) -> list[str]:
        """Queries the LLM sent to search_handbook (shortcut over tool_calls)."""
        return [
            call.args["query"]
            for call in self.tool_calls
            if call.name == "search_handbook" and "query" in call.args
        ]
