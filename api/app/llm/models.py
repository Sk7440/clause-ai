from langchain.chat_models import init_chat_model
# from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    return init_chat_model(
        model=settings.chat_model,
        model_provider=settings.chat_provider,
        api_key=settings.chat_api_key,
        # temperature=settings.chat_temperature,
    )

def get_fallback_model(settings: Settings):
    pass
