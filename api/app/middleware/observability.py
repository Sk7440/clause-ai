import logging
import time

from langchain.agents.middleware import AgentMiddleware


log = logging.getLogger("clause.agent")

class LoggingMiddleware(AgentMiddleware):
    """Logs every model call and tool call: what happened, how long, how many tokens."""
    pass
