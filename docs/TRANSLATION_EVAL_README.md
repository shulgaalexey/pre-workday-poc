# Translation Evaluation System

A LangChain-based translation quality assessment system that measures BLEU scores and fails CI/CD builds when quality drops below 50%.

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python src/evaluate_translation.py

# Run tests
pytest -v
```

## Core Components

- **`src/evaluate_translation.py`** - Main evaluation script with BLEU scoring
- **`src/agent.py`** - LangChain agent with OpenAI integration and translation tools
- **`tests/test_*.py`** - Pytest test suites with `@pytest.mark.translation_eval` markers
- **GitHub Actions** - Automated CI/CD that fails builds below 50% BLEU score

## Test Commands

```powershell
# Translation evaluation tests only
pytest -m translation_eval -v

# All tests except translation evaluation
pytest -m "not translation_eval" -v

# Run agent examples
python examples/agent_demo.py
```

## Configuration

- **BLEU Threshold**: 50% (configurable in `src/evaluate_translation.py`)
- **Test Dataset**: Spanish/German translation pairs in code
- **Environment**: Requires `OPENAI_API_KEY` for CI/CD
- **Markers**: `translation_eval`, `slow` for pytest filtering

## Expected Output

```bash
✅ BLEU score 75.0% meets threshold of 50.0%
```

Built for Windows + VS Code with clarity-first approach following PoC guidelines.
