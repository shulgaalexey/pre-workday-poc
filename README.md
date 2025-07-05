# LangChain Agent PoC

A proof-of-concept LangChain agent with OpenAI integration, built for Windows + VS Code with clarity-first approach.

## Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run
python src/agent.py
python examples/agent_demo.py
pytest -v
```

## Features

- **LangChain agent** with ReAct pattern and translation tools
- **Memory options**: In-memory (default) or persistent SQLite
- **Translation evaluation** with BLEU scoring and CI/CD integration
- **Interactive chat** and comprehensive testing

## Usage

```powershell
# Basic agent
python src/agent.py

# Examples and demos
python examples/agent_demo.py

# Translation evaluation
python src/evaluate_translation.py

# Tests (with markers)
pytest -v                           # All tests
pytest -m translation_eval -v       # Translation tests only
pytest -m "not translation_eval" -v # Skip translation tests
```

## How to Use

### 1. Basic Agent Interaction

Run the main agent for simple Q&A:

```powershell
cd c:\src\pre-workday-poc
python src/agent.py
```

This starts the agent with ReAct pattern and translation tools. Ask questions like:

- "What is the capital of France?"
- "Translate 'Hello world' to Spanish"
- "Tell me about machine learning"

### 2. Interactive Chat Interface

For a more conversational experience:

```powershell
cd c:\src\pre-workday-poc\src
python chat_cli.py
```

Features:

- Persistent conversation history
- Type `/help` for commands
- Type `/exit` to quit
- Memory persists between sessions (if configured)

### 3. Running Examples

Explore different usage patterns:

```powershell
cd c:\src\pre-workday-poc
python examples/agent_demo.py
```

This demonstrates:

- Basic agent queries
- Translation functionality
- Error handling
- Memory usage patterns

### 4. Translation Evaluation

Test translation quality with BLEU scoring:

```powershell
cd c:\src\pre-workday-poc\src
python evaluate_translation.py
```

This will:

- Run predefined translation tests
- Calculate BLEU scores
- Show results with pass/fail status
- Generate detailed evaluation reports

### 5. Running Tests

Execute the test suite:

```powershell
cd c:\src\pre-workday-poc

# Run all tests
pytest . -v --tb=short

# Run only translation evaluation tests
pytest -m translation_eval -v

# Skip translation tests (faster)
pytest -m "not translation_eval" -v

# Run specific test file
pytest tests/test_agent.py -v
```

### 6. Memory Configuration

Switch between memory types by editing `.config.yaml`:

```yaml
# For development (faster, no persistence)
memory: "in-memory"

# For production (persistent across sessions)
memory: "persistent-sqlite"
```

After changing memory type, restart the application for changes to take effect.

### 7. Environment Variables

Ensure your `.env` file contains:

```bash
OPENAI_API_KEY=your_actual_api_key_here
```

The agent requires a valid OpenAI API key to function.

### 8. Troubleshooting

**Common Issues:**

- **Import errors**: Ensure you're in the correct directory and virtual environment is activated
- **API errors**: Check your OpenAI API key is valid and has credits
- **Path issues**: Use absolute paths or ensure you're in the project root
- **Test failures**: Check if translation evaluation tests need internet connection

**Debug mode**: Set environment variable for verbose logging:

```powershell
$env:PYTHONPATH = "c:\src\pre-workday-poc"
python src/agent.py
```

## Configuration

**Memory** - Edit `.config.yaml`:

```yaml
memory: "in-memory"          # or "persistent-sqlite"
```

**Translation** - BLEU threshold in `src/evaluate_translation.py`:

```python
BLEU_THRESHOLD = 50.0        # CI/CD fails below this
```

## Project Structure

```bash
├── src/                    # Source code
│   ├── agent.py           # Main LangChain agent
│   ├── evaluate_translation.py  # Translation evaluation with BLEU scoring
│   └── chat_cli.py        # Interactive chat interface
├── tests/                 # Unit tests with pytest markers
│   ├── test_agent.py      # Agent tests
│   ├── test_evaluate_translation.py  # Translation tests
│   └── test_unicode_fix.py  # Unicode handling tests
├── docs/                  # Documentation
│   └── TRANSLATION_EVAL_README.md  # Translation evaluation docs
├── examples/
│   └── agent_demo.py      # Usage examples
├── .config.yaml          # Memory configuration
├── requirements.txt       # Dependencies
└── .env                  # OpenAI API key (create this)
```

## Translation Evaluation

The system includes automated translation quality assessment:

- **BLEU scoring** with 50% threshold
- **CI/CD integration** that fails builds on poor translations
- **Pytest markers** for selective test running
- **GitHub Actions** workflow for automated evaluation

See `docs/TRANSLATION_EVAL_README.md` for detailed translation evaluation documentation.

Built following PoC guidelines with clarity over cleverness.
