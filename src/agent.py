"""
LangChain Agent PoC

A simple proof-of-concept agent demonstrating LangChain usage with OpenAI LLM.
Built for Windows + VS Code environment with clarity-first approach.
"""

import json
import logging
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Alias for tests and to match expected OpenAI reference in tests
OpenAI = ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory (parent of src folder)."""
    import pathlib
    return pathlib.Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """
    Load configuration from .config.yaml file.

    Returns:
        Dictionary containing configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = get_project_root() / ".config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            logger.debug(f"Loaded config: {config}")
            return config or {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {"memory": "in-memory"}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        raise


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


# Load glossary once with proper encoding
try:
    glossary_path = get_project_root() / "data" / "glossary.json"
    with open(glossary_path, "r", encoding="utf-8") as f:
        GLOSS = json.load(f)
except UnicodeDecodeError as e:
    logger.error(f"Unicode error loading glossary: {e}")
    # Fallback to basic glossary
    GLOSS = {
        "cloud": {"es": "nube", "de": "Cloud"},
        "payroll": {"es": "nomina", "de": "Lohnabrechnung"},  # Simplified without accent
        "Workday": {"es": "Workday", "de": "Workday"}
    }
except Exception as e:
    logger.error(f"Error loading glossary: {e}")
    GLOSS = {}


def _apply_glossary(lang: str, text: str) -> list[dict]:
    """Return OpenAI function-calling 'glossary' argument."""
    gloss_map = {src: tgt.get(lang[:2].lower()) for src, tgt in GLOSS.items()}
    return [{"source_term": k, "target_term": v} for k, v in gloss_map.items() if v]


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
    glossary = _apply_glossary(language, text)

    logger.info(f"Translating to %s: %s", language, text)

    # Get API key - should already be available in environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return "Translation failed: OPENAI_API_KEY not found in environment variables"

    # Use LangChain ChatOpenAI for proper integration
    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.3,
        model_name="gpt-4o-mini"
    )

    # Create system message with glossary context
    glossary_context = ""
    if glossary:
        try:
            glossary_terms = ", ".join([f"{g['source_term']} -> {g['target_term']}" for g in glossary])
            glossary_context = f"\nUse these glossary terms exactly: {glossary_terms}"
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.warning(f"Unicode encoding error in glossary: {e}")
            glossary_context = "\nUse standard translation practices."

    # Enhanced prompt to ensure only the translation is returned
    system_prompt = (
        f"You are a professional translator. Translate the given text to {language}. "
        f"Return ONLY the translated text, nothing else. Do not include any explanations, "
        f"quotes, or additional text. Just the direct translation.{glossary_context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]

    try:
        response = chat_llm.invoke(messages)
        result = response.content.strip()

        # Remove any potential quotes or extra formatting
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        # Ensure proper UTF-8 encoding for Windows compatibility
        try:
            # Test encoding/decoding to catch issues early
            encoded = result.encode('utf-8')
            result = encoded.decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.warning(f"Unicode encoding issue with result: {e}")
            # Fallback to ASCII-safe version if needed
            result = result.encode('ascii', errors='replace').decode('ascii')

        logger.info(f"Translation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"Translation failed: {str(e)}"


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
    # Load configuration to determine memory type
    config = load_config()
    memory_type = config.get("memory", "in-memory")

    if memory_type == "persistent-sqlite":
        # ----- New persistent memory -----
        message_history = SQLChatMessageHistory(
            session_id="default_user",
            connection="sqlite:///chat_history.db"   # creates file in repo root
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )
        logger.info("Using persistent SQLite memory")
    else:
        # Default in-memory configuration
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("Using in-memory memory")

    # memory.save_context({"input": "You are a helpful translator"}, {"output": "Hi there! Got it."})
    logger.debug("Current memory buffer: %s", memory.buffer_as_str)

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
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
    # Load API key from .env file (for local development)
    # In CI/CD environments, the key should already be in environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("For local development, ensure you have a .env file with OPENAI_API_KEY=your_key")
        logger.error("For CI/CD, ensure OPENAI_API_KEY is set as a secret")
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
