#RAG (RETRIVAL AGUMENTED GENERATION)

#1, Ingestion: load.split,embed,store

#Collection:Shop_handbook
# point:
from langchain_core.documents import Document # .txt, .md
from langchain_community.document_loaders import PyPDFLoader # .pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain.embeddings import init_embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
DATA_PATH = "shop_handbook.pdf"

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma4:31b-cloud"
EMBEDDING_PROVIDER = "ollama"
EMBED_MODEL = "nomic-embed-text:latest"
TOP_K = 3

COLLECTION_NAME = "shop_handbook"
QDRANT_PATH = "./qdrant_data_notebook"
EMBED_DIM=768

SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)

embedd = init_embeddings(
    model=EMBED_MODEL, 
    provider=EMBEDDING_PROVIDER
)

client = QdrantClient(path=QDRANT_PATH)

# Indexing
# 1. Load
pages = PyPDFLoader(DATA_PATH).load_and_split()

# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(pages)

# 3. Embed
embedd = init_embeddings(
    model=EMBED_MODEL,
    provider=EMBEDDING_PROVIDER
)
#pre store
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print("Deleted old collection")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)
# 4. Store

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedd,
)
vector_store.add_documents(chunks)
