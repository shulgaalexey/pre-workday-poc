"""
OpenAI Configuration Module

Centralized management of OpenAI API key retrieval and validation.
Built for Windows + VS Code environment with clarity-first approach.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)


def get_openai_api_key() -> str:
    """
    Retrieve and validate OpenAI API key from environment variables.

    This function centralizes all OpenAI API key retrieval logic to ensure
    consistent behavior across the application.

    Returns:
        str: The OpenAI API key

    Raises:
        ValueError: If OPENAI_API_KEY is not found or is empty

    Note:
        - Automatically loads .env file for local development
        - Provides detailed error messages for troubleshooting
        - Supports both local development and CI/CD environments
    """
    # Load environment variables from .env file (for local development)
    # In CI/CD environments, the key should already be in environment variables
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("For local development, ensure you have a .env file with OPENAI_API_KEY=your_key")
        logger.error("For CI/CD, ensure OPENAI_API_KEY is set as a secret")
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Additional validation for empty strings
    if not openai_api_key.strip():
        logger.error("OPENAI_API_KEY is empty")
        raise ValueError("OPENAI_API_KEY is empty")

    logger.debug(f"OPENAI_API_KEY found (length: {len(openai_api_key)})")
    return openai_api_key


def check_openai_api_key() -> Optional[str]:
    """
    Check if OpenAI API key is available without raising exceptions.

    This function is useful for diagnostic purposes or when you want to
    handle the missing key gracefully.

    Returns:
        str | None: The OpenAI API key if found, None otherwise

    Note:
        - Does not raise exceptions
        - Logs appropriate messages for debugging
        - Loads .env file automatically
    """
    # Load environment variables from .env file
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key and openai_api_key.strip():
        logger.debug(f"OPENAI_API_KEY found (length: {len(openai_api_key)})")
        return openai_api_key
    else:
        logger.warning("OPENAI_API_KEY not found or is empty")
        return None


def log_api_key_debug_info() -> None:
    """
    Log detailed debug information about API key availability.

    This function is useful for troubleshooting API key issues in CI/CD
    environments or local development.

    Note:
        - Safe to call in production (doesn't expose key values)
        - Provides comprehensive diagnostic information
        - Helps identify environment variable issues
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        logger.info(f"OPENAI_API_KEY found (length: {len(openai_key)})")
    else:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Available environment variables:")
        for key in sorted(os.environ.keys()):
            if 'API' in key or 'OPENAI' in key:
                value = os.environ[key]
                logger.error(f"  {key}: {'SET' if value else 'EMPTY'}")
