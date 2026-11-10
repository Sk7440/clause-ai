from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from pydantic import BaseModel,StringConstraints
from typing import Annotated
from app.services.chat import RAGServices,Answer

@asynccontextmanager
async def lifespan(app: FastAPI)->None:
    app.state.rag=RAGServices()
    yield
    app.state.rag.close()

app=FastAPI("Clause AI APIs")

class AskQuestion(BaseModel):
    question:Annotated[
        str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)
    ]


@app.get("/")
def health()->dict[str,str]:
    return {"status":"ok"}
@app.post("/ask",response_model=Answer)
def ask_query(body:AskQuestion,req:Request):
    return req.app.state.rag.ask(body.question)