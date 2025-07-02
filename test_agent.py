"""
Test module for agent.py

Tests the LangChain agent functionality following pytest best practices.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent import create_langchain_agent, echo_tool


class TestEchoTool:
    """Test cases for the echo_tool function."""

    def test_echo_tool_basic(self):
        """Test basic echo functionality."""
        test_input = "Hello, World!"
        result = echo_tool(test_input)
        assert result == f"Echo: {test_input}"

    def test_echo_tool_empty_string(self):
        """Test echo with empty string."""
        result = echo_tool("")
        assert result == "Echo: "

    def test_echo_tool_special_characters(self):
        """Test echo with special characters."""
        test_input = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = echo_tool(test_input)
        assert result == f"Echo: {test_input}"


class TestAgentCreation:
    """Test cases for agent creation and configuration."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('agent.create_react_agent')
    @patch('agent.OpenAI')
    def test_create_agent_success(self, mock_openai, mock_create_react_agent):
        """Test successful agent creation with valid API key."""
        # Mock the LLM and agent
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent

        # Create agent
        agent_executor = create_langchain_agent()

        # Verify the agent was created and called
        assert agent_executor is mock_agent
        mock_openai.assert_called_once()
        mock_create_react_agent.assert_called_once()

    @patch('agent.load_dotenv')
    @patch.dict(os.environ, {}, clear=True)
    def test_create_agent_missing_api_key(self, mock_load_dotenv):
        """Test agent creation fails without API key."""
        # Ensure the environment is clear
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            create_langchain_agent()

    @patch('agent.load_dotenv')
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_create_agent_empty_api_key(self, mock_load_dotenv):
        """Test agent creation fails with empty API key."""
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            create_langchain_agent()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
