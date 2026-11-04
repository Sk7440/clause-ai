from langchain_core import vectorstores
from pathlib import Path


from langchain.chat_models import init_chat_model 
from langchain.embeddings import  init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore



#Indexing
#Load
doc = Document(
    page_content="LangChain is a framework for building LLM-powered applications.",
    metadata={"source": "intro_guide", "author": "LangChain"},
    id="doc_1",
)


#Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # Maximum number of characters per chunk
    chunk_overlap=100,     # Number of characters overlapping between adjacent chunks
    separators=["\n\n", "\n", " ", ""],
)
chunks = text_splitter.split_documents([doc])


#EMBED
embeddings = init_embeddings(
    model="nomic-embed-text:latest",
    provider="ollama"
)
# Store

vector_store=InMemoryVectorStore.from_documents(
    documents=chunks,
embedding=embeddings
) 

#RAG
#LLM
llm=init_chat_model(
    model="gemma4:31b-cloud",
    model_provider="ollama",
    temperature=0.1
)
query=input("query:").strip()

#Retrive
retriver=vector_store.similarity_search(query,3)

texts=[]
for doc in retriver:
    texts.append(doc.page_content)
context="\n\n".join(texts)

#Generate
messages=[
    {"role":"system","content":"Answer only from the context. If the answer isn’t in the context, say you don’t know."},
    {
"role":"user","content":f"context:{context}\n\nQuestion:{query}"
    }
]
response = llm.invoke(messages)
print(response.content)

