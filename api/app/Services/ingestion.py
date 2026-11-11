from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from app.core.config import get_settings, get_embeddings, get_qdrant_client


def load(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def ingest(path: str | None = None) -> int:
    settings = get_settings()
    file = Path(path or settings.data_path)
    
    if not file.exists():
        raise FileNotFoundError(f"PDF file not found at: {file.resolve()}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(load(path=file))

    embed = get_embeddings(settings)
    client = get_qdrant_client(settings)
    
    try:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)

        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
        )
        vector_store.add_documents(chunks)
        print(f"Successfully ingested {len(chunks)} chunks into '{settings.collection_name}'.")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise
    finally:
        client.close()

    return len(chunks)


if __name__ == "__main__":
    ingest()