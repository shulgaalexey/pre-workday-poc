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
python agent.py
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
python agent.py

# Examples and demos
python examples/agent_demo.py

# Translation evaluation
python evaluate_translation.py

# Tests (with markers)
pytest -v                           # All tests
pytest -m translation_eval -v       # Translation tests only
pytest -m "not translation_eval" -v # Skip translation tests
```

## Configuration

**Memory** - Edit `.config.yaml`:

```yaml
memory: "in-memory"          # or "persistent-sqlite"
```

**Translation** - BLEU threshold in `evaluate_translation.py`:

```python
BLEU_THRESHOLD = 50.0        # CI/CD fails below this
```

## Project Structure

```bash
├── agent.py                 # Main LangChain agent
├── evaluate_translation.py  # Translation evaluation with BLEU scoring
├── chat_cli.py             # Interactive chat interface
├── test_*.py               # Unit tests with pytest markers
├── examples/agent_demo.py  # Usage examples
├── .config.yaml           # Memory configuration
├── requirements.txt        # Dependencies
└── .env                   # OpenAI API key (create this)
```

## Translation Evaluation

The system includes automated translation quality assessment:

- **BLEU scoring** with 50% threshold
- **CI/CD integration** that fails builds on poor translations
- **Pytest markers** for selective test running
- **GitHub Actions** workflow for automated evaluation

See `TRANSLATION_EVAL_README.md` for detailed translation evaluation documentation.

Built following PoC guidelines with clarity over cleverness.
