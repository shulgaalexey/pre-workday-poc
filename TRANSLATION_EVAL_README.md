# Translation Evaluation Documentation

This documentation describes the translation evaluation system implemented for the pre-workday-poc project.

## Overview

The translation evaluation system uses BLEU scores to assess translation quality and automatically fails builds if the score falls below the threshold (50%).

## Components

### 1. evaluate_translation.py
Main evaluation module that:
- Runs translation evaluation on test datasets
- Calculates BLEU scores using exact match approximation
- Exits with appropriate codes for CI/CD integration

### 2. test_evaluate_translation.py
Pytest test suite with `@pytest.mark.translation_eval` markers that:
- Tests BLEU score threshold enforcement
- Validates failure mechanisms
- Provides comprehensive test coverage

### 3. GitHub Actions Workflow
Automated CI/CD pipeline (`.github/workflows/translation-eval.yml`) that:
- Runs on every push to main/develop branches
- Executes translation evaluation
- Fails builds if BLEU score < 50%

## Usage

### Manual Evaluation
```powershell
# Run translation evaluation
python evaluate_translation.py

# Expected output:
# ✅ BLEU score 75.0% meets threshold of 50.0%
# (or)
# ❌ BLEU score 25.0% below threshold of 50.0%
```

### Running Tests
```powershell
# Run only translation evaluation tests
pytest -m translation_eval -v

# Run all tests except translation evaluation
pytest -m "not translation_eval" -v

# Run all tests
pytest -v
```

### Installing Dependencies
```powershell
# Install all requirements including evaluation dependencies
pip install -r requirements.txt
```

## Configuration

### BLEU Threshold
The BLEU threshold is set to 50% in `evaluate_translation.py`:
```python
BLEU_THRESHOLD = 50.0
```

### Test Dataset
Current test dataset includes:
- Spanish: "cloud payroll" → "nube nómina"
- German: "Workday payroll" → "Workday Lohnabrechnung"

### Pytest Markers
- `@pytest.mark.translation_eval`: Translation evaluation tests
- `@pytest.mark.slow`: Slow-running tests

## CI/CD Integration

The GitHub Actions workflow:
1. Sets up Python 3.11 environment
2. Installs dependencies
3. Runs `python evaluate_translation.py`
4. Executes pytest with translation_eval marker
5. Uploads results as artifacts

### Environment Variables
Required for CI/CD:
- `OPENAI_API_KEY`: OpenAI API key for agent functionality

## Production Considerations

For production deployment, consider:
- Using proper BLEU libraries (sacrebleu)
- Implementing COMET for neural evaluation metrics
- Expanding test datasets
- Adding more comprehensive evaluation metrics

## Error Handling

The system includes comprehensive error handling:
- Graceful handling of empty evaluation results
- Proper logging of errors and warnings
- Appropriate exit codes for CI/CD systems

## Example Output

```
2025-07-03 10:30:15 - evaluate_translation - INFO - Starting translation evaluation
2025-07-03 10:30:16 - evaluate_translation - INFO - Agent created successfully
2025-07-03 10:30:16 - evaluate_translation - INFO - Running evaluation on 2 test cases
2025-07-03 10:30:18 - evaluate_translation - INFO - Evaluation completed successfully
2025-07-03 10:30:18 - evaluate_translation - INFO - Calculated BLEU score: 100.0% (2/2 exact matches)
✅ BLEU score 100.0% meets threshold of 50.0%
```
