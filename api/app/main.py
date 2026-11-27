from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.schemas.agent import AgentReply, AskRequest
from app.services.agent import ClauseAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = ClauseAgent()
    yield
    app.state.agent.close()

app = FastAPI(title="Clause AI APIs", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AgentReply)
def ask_query(body: AskRequest, req: Request) -> AgentReply:
    return req.app.state.agent.ask(body.question,body.thread_id)


# cd api
# uv run fastapi dev app/main.py
