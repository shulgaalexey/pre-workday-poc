"""
Simple LangChain Agent Example

Demonstrates basic usage of the LangChain agent with different input types.
This is experimental code for testing agent behavior.
"""

import logging
import os
import sys

# Add parent directory to path to import agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import create_langchain_agent

# Set up logging for the example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_agent_examples():
    """
    Run the agent with various example inputs to test behavior.
    """
    try:
        # Create the agent
        agent = create_langchain_agent()

        # Example inputs to test
        test_inputs = [
            "What is 2 + 2?",
            "Echo back: Hello from the agent!",
            "Can you help me with a simple task?",
            "Tell me about yourself"
        ]

        print("=" * 60)
        print("LangChain Agent Examples")
        print("=" * 60)

        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n--- Example {i} ---")
            print(f"Input: {user_input}")

            try:
                result = agent.invoke({"input": user_input})
                output = result.get("output", "No output generated")
                print(f"Output: {output}")

            except Exception as e:
                logger.error(f"Error in example {i}: {e}")
                print(f"Error: {e}")

            print("-" * 40)

        print("\nExamples completed!")

    except Exception as e:
        logger.error(f"Failed to create or run agent: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    run_agent_examples()
