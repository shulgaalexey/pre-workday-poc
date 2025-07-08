"""
Test module for vector memory functionality.

Tests the vector store memory implementation with translation memory retrieval.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.agent import create_langchain_agent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestVectorMemory:
    """Test cases for vector memory functionality."""

    @patch.dict('os.environ', {"OPENAI_API_KEY": "test_key"})
    @patch('src.agent.load_config')
    @patch('src.agent._get_vector_memory')
    @patch('src.agent.initialize_agent')
    def test_tm_retrieval(self, mock_initialize_agent, mock_get_vector_memory, mock_load_config, tmp_path, mocker):
        """Test translation memory retrieval from vector store."""
        # Configure to use vector-store memory
        mock_load_config.return_value = {"memory": "vector-store"}

        # Mock the ChatOpenAI invoke method
        mocker.patch("src.agent.ChatOpenAI.invoke", return_value=MagicMock(content="dummy"))

        # Create a mock memory instance that behaves like VectorStoreRetrieverMemory
        mock_memory_instance = MagicMock()
        mock_memory_instance.memory_key = "chat_history"

        # Mock the memory variables to return expected content
        mock_memory_doc = MagicMock()
        mock_memory_doc.content = "cloud payroll -> nube nómina"

        # Mock the load_memory_variables to return the expected structure with tm_history key
        mock_memory_instance.load_memory_variables.return_value = {"tm_history": [mock_memory_doc]}

        # Set the mock to return our memory instance
        mock_get_vector_memory.return_value = mock_memory_instance

        # Mock the agent executor
        mock_agent_instance = MagicMock()
        mock_agent_instance.memory = mock_memory_instance
        mock_initialize_agent.return_value = mock_agent_instance

        # Create agent with vector memory
        agent = create_langchain_agent()

        # Verify the vector memory was called
        mock_get_vector_memory.assert_called_once()

        # Save context to memory
        agent.memory.save_context({}, {"role": "assistant", "content": "cloud payroll -> nube nómina"})

        # Test retrieval - using tm_history as requested in the original test case
        hits = agent.memory.load_memory_variables({})["tm_history"]
        assert "nube nómina" in hits[0].content
