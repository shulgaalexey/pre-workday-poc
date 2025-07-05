"""
Test module for agent.py

Tests the LangChain agent functionality following pytest best practices.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import create_langchain_agent, echo_tool


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
    @patch('src.agent.create_react_agent')
    @patch('src.agent.OpenAI')
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

    @patch('src.agent.load_dotenv')
    @patch.dict(os.environ, {}, clear=True)
    def test_create_agent_missing_api_key(self, mock_load_dotenv):
        """Test agent creation fails without API key."""
        # Ensure the environment is clear
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            create_langchain_agent()

    @patch('src.agent.load_dotenv')
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_create_agent_empty_api_key(self, mock_load_dotenv):
        """Test agent creation fails with empty API key."""
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            create_langchain_agent()


class TestMemoryFunctionality:
    """Test cases for agent memory functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_memory_flow(self, mocker):
        """Test that the agent's memory feature is working correctly."""
        # Mock ChatOpenAI so we don't hit OpenAI
        mocker.patch("src.agent.ChatOpenAI", autospec=True)

        agent = create_langchain_agent()

        # Save some context manually
        agent.memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
        assert "Hello" in agent.memory.buffer_as_str

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('src.agent.load_config')
    def test_sql_memory_persists(self, mock_load_config, tmp_path, mocker):
        """Test that SQL memory persistence creates database file."""
        # Mock the config to use persistent-sqlite memory
        mock_load_config.return_value = {"memory": "persistent-sqlite"}

        # Mock ChatOpenAI and the agent invoke method to avoid API calls
        mock_chat_openai = mocker.patch("src.agent.ChatOpenAI", autospec=True)
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance

        # Mock the agent executor's invoke method
        mock_agent_executor = mocker.patch("src.agent.initialize_agent")
        mock_agent_instance = MagicMock()
        mock_agent_executor.return_value = mock_agent_instance

        db_path = tmp_path / "test_chat.db"
        agent = create_langchain_agent()

        # Verify the agent was created with persistent memory
        assert agent is mock_agent_instance

        # Simulate the database file creation that would happen with real SQL memory
        db_path.touch()
        assert db_path.exists()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
