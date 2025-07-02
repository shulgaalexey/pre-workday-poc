# LangChain Agent PoC

A simple proof-of-concept demonstrating LangChain agent functionality with OpenAI LLM integration.
Built for Windows + VS Code environment following clarity-first development principles.

## Features

- Modern LangChain agent implementation using ReAct pattern
- OpenAI LLM integration with proper error handling
- Comprehensive logging for debugging and monitoring
- Unit tests with pytest for reliability
- Example scripts for experimentation

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone the repository and navigate to the project folder:**

   ```powershell
   cd c:\src\pre-workday-poc
   ```

2. **Create and activate a virtual environment:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Basic Agent

   ```powershell
   python .\agent.py
   ```

### Running Examples

   ```powershell
   python .\examples\agent_demo.py
   ```

### Running Tests

   ```powershell
   # Run all tests
   python -m pytest test_agent.py -v

   # Run with coverage
   python -m pytest test_agent.py --cov=agent --cov-report=html
   ```

## Project Structure

```text
├── agent.py              # Main LangChain agent implementation
├── test_agent.py         # Unit tests for agent functionality
├── examples/
│   └── agent_demo.py     # Example usage demonstrations
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md            # This file
```

## Architecture

The project follows a simple, modular architecture:

- **`agent.py`**: Core agent implementation with proper error handling and logging
- **`echo_tool`**: Simple demonstration tool that echoes input text
- **`create_langchain_agent()`**: Factory function for agent creation and configuration
- **Tests**: Comprehensive unit tests covering functionality and edge cases

## Development Guidelines

Following the repository's coding principles:

1. **Clarity over cleverness** - straightforward, readable code
2. **Comprehensive logging** - INFO level by default, DEBUG for detailed diagnostics
3. **Type hints and docstrings** - clear documentation for all functions
4. **Unit testing** - pytest with ≥80% coverage target
5. **PoC mindset** - rapid experimentation with small, focused scripts

## Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: No module named 'langchain_community'`**
   - Solution: Install missing dependencies with `pip install langchain-openai langchain-community`

2. **`ValueError: OPENAI_API_KEY not found`**
   - Solution: Ensure your `.env` file contains a valid OpenAI API key

3. **Agent parsing errors**
   - This is expected behavior with the current ReAct prompt template
   - The agent will handle parsing errors gracefully and continue execution

### Logging

To increase verbosity:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Contributing

1. Follow the existing code style and patterns
2. Add unit tests for new functionality
3. Update documentation for significant changes
4. Use descriptive commit messages

## License

This is a proof-of-concept project for internal use.
