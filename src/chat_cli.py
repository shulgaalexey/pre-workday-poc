import sys

from agent import create_langchain_agent, logger


def main():
    agent = create_langchain_agent()
    print("Type 'exit' to quit.\n")
    while True:
        # Only log buffer contents for memory types that support it
        if hasattr(agent.memory, 'buffer_as_str'):
            logger.info("Current memory buffer:\n%s", agent.memory.buffer_as_str)
        else:
            logger.info("Memory type: %s", type(agent.memory).__name__)

        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Good-bye!")
            break
        result = agent.invoke({"input": user_input})
        print("Agent:", result["output"])

if __name__ == "__main__":
    main()
