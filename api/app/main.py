from services.chat import RAGService
from services.ingestion import ingest


def main():
    rag_service=RAGService()
    try:
        query=input("you:").strip()
        result= rag_service.ask(query)
        print(f"AI:{result.answer}")
    except:
            pass
    finally:
        rag_service.close()


if __name__=="__main__":
    #ingest()
    main()