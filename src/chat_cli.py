import sys

from agent import create_langchain_agent, logger


def main():
    agent = create_langchain_agent()
    print("Type 'exit' to quit.\n")
    while True:
        logger.info("Current memory buffer:\n%s", agent.memory.buffer_as_str)

        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Good-bye!")
            break
        result = agent.invoke({"input": user_input})
        print("Agent:", result["output"])

if __name__ == "__main__":
    main()
