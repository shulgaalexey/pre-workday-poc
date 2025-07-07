"""
Vector Memory Example

Demonstrates the vector store memory functionality for semantic retrieval.
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_temp_config_with_vector_memory():
    """Create a temporary config file with vector-store memory."""
    config_content = """# Temporary configuration for vector memory demo
memory: "vector-store"
"""
    config_path = Path(__file__).parent.parent / ".config.yaml.backup"

    # Backup existing config if it exists
    main_config = Path(__file__).parent.parent / ".config.yaml"
    if main_config.exists():
        main_config.rename(config_path)

    # Create new config with vector memory
    with open(main_config, 'w', encoding='utf-8') as f:
        f.write(config_content)

    return config_path

def restore_config(backup_path):
    """Restore the original config file."""
    main_config = Path(__file__).parent.parent / ".config.yaml"
    if backup_path.exists():
        backup_path.rename(main_config)
    elif main_config.exists():
        main_config.unlink()

def demo_vector_memory():
    """
    Demonstrate vector memory functionality.

    This example shows how the agent uses FAISS vector store for semantic retrieval.
    """
    backup_path = None

    try:
        # Set up environment for demo
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not found. This demo won't work without it.")
            logger.info("Please set your OpenAI API key in a .env file or environment variable.")
            return

        # Create temporary config
        backup_path = create_temp_config_with_vector_memory()

        # Import and create agent with vector memory
        from src.agent import create_langchain_agent

        logger.info("Creating agent with vector store memory...")
        agent = create_langchain_agent()

        # Test inputs that demonstrate semantic retrieval
        test_inputs = [
            "Spanish | Hello world",
            "German | Good morning",
            "French | Thank you",
            "Can you translate 'goodbye' to Spanish?",
            "What was the first translation I asked for?"
        ]

        logger.info("Running demo interactions...")
        for i, user_input in enumerate(test_inputs, 1):
            logger.info(f"\n--- Interaction {i} ---")
            logger.info(f"User: {user_input}")

            try:
                result = agent.invoke({"input": user_input})
                logger.info(f"Agent: {result.get('output', 'No output generated')}")
            except Exception as e:
                logger.error(f"Error in interaction {i}: {e}")

        logger.info("\nVector memory demo completed successfully!")
        logger.info("The agent used FAISS vector store for semantic retrieval of conversation history.")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        # Restore original config
        if backup_path:
            restore_config(backup_path)
            logger.info("Original configuration restored.")

if __name__ == "__main__":
    demo_vector_memory()
