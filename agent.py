"""
LangChain Agent PoC

A simple proof-of-concept agent demonstrating LangChain usage with OpenAI LLM.
Built for Windows + VS Code environment with clarity-first approach.
"""

import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Alias for tests and to match expected OpenAI reference in tests
OpenAI = ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def echo_tool(text: str) -> str:
    """
    Simple echo tool that returns the same text back.

    Args:
        text: Input text to echo back

    Returns:
        The same input text
    """
    logger.debug(f"Echo tool called with: {text}")
    return f"Echo: {text}"


def translate_tool(input_text: str) -> str:
    """
    Translate text to a specified language.

    Args:
        input_text: In format '<language> | <text>' e.g. 'Russian | Hello'

    Returns:
        Translated text or error message.
    """
    parts = input_text.split("|", 1)
    if len(parts) != 2:
        return f"Error: Invalid format. Use '<language> | <text>'. Got: {input_text}"
    language, text = parts[0].strip(), parts[1].strip()
    logger.debug(f"Translating to %s: %s", language, text)
    chat_llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3, model_name="gpt-4")
    messages = [
        SystemMessage(content=f"Translate the following text to {language}. Return only the translated text."),
        HumanMessage(content=text)
    ]
    logger.info(f"\nSending translation request to OpenAI LLM: {input_text}")
    response = chat_llm(messages)
    return response.content.strip()


# New function to initialize the ReAct agent for testability and separation of concerns
def create_react_agent(llm: Any, tools: list) -> Any:
    """
    Initialize a zero-shot ReAct agent with provided LLM and tools.

    Args:
        llm: Language model instance
        tools: List of tools for the agent

    Returns:
        An initialized agent instance
    """
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )


def create_langchain_agent() -> AgentExecutor:
    """
    Create and configure a LangChain agent with OpenAI LLM and tools.

    Returns:
        Configured AgentExecutor instance

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment
    """
    # Load API key from .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    logger.info("Initializing LangChain agent with OpenAI LLM")

    # Initialize the chat LLM wrapper for chat endpoint using alias for testability
    llm = OpenAI(
        openai_api_key=openai_api_key,
        temperature=0.7,
        model_name="gpt-4"  # Use GPT-4 chat model via v1/chat/completions
    )

    # Define tools available to the agent
    tools = [
        Tool(
            name="Echo",
            func=echo_tool,
            description="Echoes input text back to the user."
        ),
        Tool(
            name="Translate",
            func=translate_tool,
            description="Translates text. Input format: '<language> | <text>' e.g. 'Russian | Hello'."
        )
    ]

    # Initialize and return the AgentExecutor via create_react_agent
    logger.info("Initializing AgentExecutor via create_react_agent")
    agent_executor = create_react_agent(llm, tools)

    logger.info("Agent successfully initialized")
    return agent_executor


def main() -> None:
    """
    Main function to demonstrate the LangChain agent functionality.
    """
    try:
        # Create the agent
        agent = create_langchain_agent()

        # Run the agent on a sample input
        # user_input = "Hello, LangChain agent!"
        user_input = "Russian | Good morning!"
        logger.info(f"Running agent with input: {user_input}")

        result = agent.invoke({"input": user_input})

        print("Agent response:", result.get("output", "No output generated"))
        logger.info("Agent execution completed successfully")

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise


if __name__ == "__main__":
    main()
