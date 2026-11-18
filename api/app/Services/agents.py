from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from app.services.chat import RAGServices
from langchain.tools import tool

# python Classes
#inheritance -- strong relationship --- is a
# Association -- weak relationship has a (1. composition , 2. aggregation)
class ClauseAgent:
    def __init__(self)-> None:

        self.rag = RAGServices() #instance attribute classs type
        
        @tool(response_format="content_and_artifact")
        def search_handbook(question:str) -> None:
            """this function"""
            results=self.rag.store.similarity_search_with_score(
                query=question,
                k=self.rag.settings.top_k
            )
            if not results:
            return "nothing found",[]

            text=[]
            sources=[]
            fo doc,score in results:
            text.append(doc.page_content)
            sources.append({"sources": doc.metadat.get("sources"),
            "page": doc.metadat.get("page"),
            "score" : round (score,4)})
            text= "\n\n".join(texts)
            return text,sources

        self.agent = create_agent(
            model = self.rag.llm,
            tools = [],
            system_prompt=self.rag.settings.system_prompt
            
        )
    
    def ask(self,question:str)-> None:
        result =self.agent.invoke({"messages": [HumanMessage(question)]})
        messages=result [" messages"]
        answer = messages[-1].text
        return answer
    
    def close(self)->None:
        self.rag.close()
        





# conversation loop
# Agentic loop




# llm=init_chat_model(
#     model="gemma4:31b-cloud",
#     tools=[],
#     system_prompt="you are helpful clause ai assistant"

# )

# query=input("you :")

# messages-=[
#     {"role": "user", "content": query}
# ]

# result