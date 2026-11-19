# Clause AI

Clause AI is a retrieval-augmented assistant for answering questions about the
shop handbook. The application loads the handbook PDF, splits it into chunks,
indexes the chunks in a local Qdrant database, and uses an LLM to answer
questions with relevant handbook sources.

The runnable application is in [`api/`](./api).
The `playground/` and `notebook/` directories contain experiments and learning
materials.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- Ollama models:

  ```powershell
  ollama pull gemma3:1b
  ollama pull nomic-embed-text:latest
  ```

The default configuration uses Ollama for both chat and embeddings, so no
cloud API key is required.

## Setup

From the repository root, enter the API project and install its dependencies:

```powershell
cd api
uv sync
```

Create `api/.env` if you need to override the defaults. Settings use the
`RAG_` prefix:

```dotenv
RAG_CHAT_PROVIDER=ollama
RAG_CHAT_MODEL=gemma3:1b
RAG_EMBED_PROVIDER=ollama
RAG_EMBED_MODEL=nomic-embed-text:latest
RAG_EMBED_DIM=768
RAG_DATA_PATH=files/shop_handbook.pdf
RAG_QDRANT_PATH=./vector_stores
RAG_COLLECTION_NAME=shop_handbook
RAG_TOP_K=3
RAG_SYSTEM_PROMPT_VERSION=v1
```

`api/.env` is ignored by Git. Do not commit API keys or other secrets.

## Index the handbook

Run ingestion from the `api` directory:

```powershell
uv run python -m app.services.ingestion
```

This command reads `api/files/shop_handbook.pdf`, creates chunks, generates
embeddings, and recreates the local Qdrant collection. Run it again whenever
the handbook or chunking settings change.

## Run the API

Start the development server from the `api` directory:

```powershell
uv run fastapi dev app/main.py
```

The server is available at <http://127.0.0.1:8000>. Useful endpoints:

- `GET /` — health check
- `POST /ask` — ask a handbook question
- `/docs` — interactive Swagger API documentation

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What is the refund policy?"}'
```

The response contains the generated answer and the handbook sources used to
produce it.

## Run the CLI

For a one-question terminal interaction:

```powershell
uv run python -m app.cli
```

Enter a question at the `You:` prompt. Press `Ctrl+C` or send EOF to exit.

## Run evaluations

The evaluation runner asks the golden questions in `api/evals/` and writes a
timestamped JSON report to `api/evals/results/`:

```powershell
uv run python -m evals.run
```

The handbook must already be indexed before running evaluations.

## Project layout

```text
api/
  app/
    main.py                 FastAPI application
    cli.py                  Terminal entry point
    services/ingestion.py   PDF ingestion and indexing
    services/agents.py      Clause agent and handbook search tool
    core/config.py          Environment-backed settings
    rag/vector_store.py     Local Qdrant vector store
    prompts/                System prompts
  files/shop_handbook.pdf   Source handbook
  evals/                    Golden questions and evaluation runner
  vector_stores/            Local Qdrant data
playground/                 RAG experiments
notebook/                   Jupyter notebooks
```

## Troubleshooting

- **Collection not found:** run the ingestion command again.
- **Ollama connection errors:** make sure Ollama is running and both required
  models have been pulled.
- **Embedding dimension error:** ensure `RAG_EMBED_DIM` matches the dimensions
  returned by the configured embedding model.
- **Wrong handbook path:** run commands from `api`, or set
  `RAG_DATA_PATH` to the correct path.