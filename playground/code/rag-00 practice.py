from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# --- Load: a real document with several pages of text ---
file = Path("notes/claude_api_notes.txt")          # any long .txt file of your notes
doc = Document(
    page_content=file.read_text(encoding="utf-8"),
    metadata={"source": file.name},
)

# --- Chunk ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents([doc])
print(f"{len(chunks)} chunks")                      # should be more than 1

# --- Embed + store ---
embeddings = init_embeddings(model="nomic-embed-text:latest", provider="ollama")
vector_store = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)

# --- Model ---
llm = init_chat_model(model="gemma4:31b-cloud", model_provider="ollama", temperature=0.1)

query = input("query: ").strip()
if not query:
    raise SystemExit("Please type a question.")

# --- Retrieve, and show what came back ---
results = vector_store.similarity_search_with_score(query, k=3)
print("\nRetrieved:")
for i, (chunk, score) in enumerate(results, start=1):
    print(f"[{i}] score={score:.3f} | {chunk.page_content[:80]}...")

context = "\n\n".join(
    f"[{i}] (source: {chunk.metadata['source']})\n{chunk.page_content}"
    for i, (chunk, score) in enumerate(results, start=1)
)

# --- Generate ---
messages = [
    {"role": "system", "content": (
        "Answer only from the context. If the answer isn't in the context, say you don't know. "
        "Cite the chunk numbers you used, like [1]."
    )},
    {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}"},
]
response = llm.invoke(messages)
print("\nAnswer:\n", response.text)