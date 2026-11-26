from app.services.agent import ClauseAgent


def main():
    clause_agent = ClauseAgent()
    try:
        thread_id =None
        while True:
            question = input("You: ").strip()
            if not question:
                continue

            reply = clause_agent.ask(question)
            thread_id=reply.thread_id
            print(f"AI: {reply.answer}")
            print(f"--- searched: {reply.searches}")
            print(f"--- pages:    {[s.page for s in reply.sources]}")

            # state = clause_agent.agent.get_state({"configurable": {"thread_id": thread_id}})
            # print(f"--- memory:     {len(state.values["messages"])} messages")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        clause_agent.close()


if __name__ == "__main__":
    main()

# cd api
# uv run python -m app.cli
