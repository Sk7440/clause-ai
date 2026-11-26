from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    SummarizationMiddleware,
    ClearToolUsesEdit, ContextEditingMiddleware
)

from app.core.config import Settings
from app.llm.models import get_chat_model, get_fallback_model
from app.guardrails.pii import pii_guardrails


def default_stack(settings: Settings) -> list[AgentMiddleware]:
    stack = [
        ModelCallLimitMiddleware(
            run_limit=settings.max_model_calls, exit_behavior="end"
        ),
        ToolCallLimitMiddleware(
            run_limit=settings.max_tool_calls
        ),
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=2,
                    # placeholder="[old search result removed]",
                ),
            ],
        ),
        SummarizationMiddleware(
            model=get_chat_model(settings),
            trigger=("tokens", settings.summary_trigger_tokens),
            keep=("messages", settings.summary_keep_messages)
        )
    ]

    fallback = get_fallback_model(settings)
    if fallback:
        stack.append(ModelFallbackMiddleware(fallback))

    return stack
