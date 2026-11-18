from app.services.agent import ClauseAgent


def main():
    clause_agent = ClauseAgent()
    try:
        query = input("You: ").strip()
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


# cd api
# uv run python -m app.cli