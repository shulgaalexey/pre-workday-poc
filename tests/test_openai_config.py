"""
Test module for openai_config.py

Tests the centralized OpenAI API key management functionality.
"""

import os
from unittest.mock import patch

import pytest

from src.openai_config import (check_openai_api_key, get_openai_api_key,
                               log_api_key_debug_info)


class TestGetOpenAIAPIKey:
    """Test cases for the get_openai_api_key function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key_12345"})
    @patch('src.openai_config.load_dotenv')
    def test_get_api_key_success(self, mock_load_dotenv):
        """Test successful API key retrieval."""
        mock_load_dotenv.return_value = None

        api_key = get_openai_api_key()

        assert api_key == "test_api_key_12345"
        mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.openai_config.load_dotenv')
    def test_get_api_key_missing(self, mock_load_dotenv):
        """Test API key retrieval when key is missing."""
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found in environment variables"):
            get_openai_api_key()

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    @patch('src.openai_config.load_dotenv')
    def test_get_api_key_empty(self, mock_load_dotenv):
        """Test API key retrieval when key is empty."""
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found in environment variables"):
            get_openai_api_key()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "   "})
    @patch('src.openai_config.load_dotenv')
    def test_get_api_key_whitespace_only(self, mock_load_dotenv):
        """Test API key retrieval when key contains only whitespace."""
        mock_load_dotenv.return_value = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY is empty"):
            get_openai_api_key()


class TestCheckOpenAIAPIKey:
    """Test cases for the check_openai_api_key function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key_67890"})
    @patch('src.openai_config.load_dotenv')
    def test_check_api_key_success(self, mock_load_dotenv):
        """Test successful API key check."""
        mock_load_dotenv.return_value = None

        api_key = check_openai_api_key()

        assert api_key == "test_api_key_67890"
        mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.openai_config.load_dotenv')
    def test_check_api_key_missing(self, mock_load_dotenv):
        """Test API key check when key is missing."""
        mock_load_dotenv.return_value = None

        api_key = check_openai_api_key()

        assert api_key is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    @patch('src.openai_config.load_dotenv')
    def test_check_api_key_empty(self, mock_load_dotenv):
        """Test API key check when key is empty."""
        mock_load_dotenv.return_value = None

        api_key = check_openai_api_key()

        assert api_key is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "   "})
    @patch('src.openai_config.load_dotenv')
    def test_check_api_key_whitespace_only(self, mock_load_dotenv):
        """Test API key check when key contains only whitespace."""
        mock_load_dotenv.return_value = None

        api_key = check_openai_api_key()

        assert api_key is None


class TestLogAPIKeyDebugInfo:
    """Test cases for the log_api_key_debug_info function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "OTHER_API_KEY": "other_value"})
    @patch('src.openai_config.load_dotenv')
    def test_log_debug_info_with_api_key(self, mock_load_dotenv, caplog):
        """Test logging debug info when API key is present."""
        mock_load_dotenv.return_value = None

        log_api_key_debug_info()

        # Check that appropriate log messages are generated
        assert "OPENAI_API_KEY found" in caplog.text
        mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {"OTHER_API_KEY": "other_value", "SOME_OPENAI_CONFIG": "config"}, clear=True)
    @patch('src.openai_config.load_dotenv')
    def test_log_debug_info_without_api_key(self, mock_load_dotenv, caplog):
        """Test logging debug info when API key is missing."""
        mock_load_dotenv.return_value = None

        log_api_key_debug_info()

        # Check that appropriate error messages are generated
        assert "OPENAI_API_KEY not found" in caplog.text
        assert "Available environment variables" in caplog.text


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
