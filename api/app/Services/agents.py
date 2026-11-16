from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langcahin.m

llm=init_chat_model(
    model="gemma4:31b-cloud",
    tools=[],
    system_prompt="you are helpful clause ai assistant"

)

query=input("you :")

messages-=[
    {"role": "user", "content": query}
]

result