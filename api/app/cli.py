
from app.Services.ingestion import ingest
from app.Services.chat import RAGService


def main():
    rag_service = RAGService()

    try:
        query = input("You: ").strip()

        if not query:
            print("Please enter a question.")
            return

        result = rag_service.ask(query)
        print(f"AI: {result.answer}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        rag_service.close()


if __name__ == "__main__":
    # ingest()
    main()
