from app.services.agents import ClauseAgent


def main() -> None:
    clause_agent = ClauseAgent()
    try:
        while True:
            query = input("You: ").strip()
        if not query:
            return
        result = clause_agent.ask(query)
        print(f"AI: {result}")


        
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        clause_agent.close()


if __name__ == "__main__":
    main()