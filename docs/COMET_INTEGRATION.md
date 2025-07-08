# COMET Score Integration

This document summarizes the changes made to integrate COMET (Crosslingual Optimized Metric for Evaluation of Translation) scoring into the translation evaluation module.

## Overview

The `evaluate_translation.py` module has been enhanced to include:

- COMET neural evaluation metrics via `unbabel-comet`
- Improved BLEU calculation via `sacrebleu`
- Quality gate checks for CI/CD
- Structured results output with timestamps
- Comprehensive test coverage

## Changes Made

### 1. Dependencies Updated (`requirements.txt`)

Added new dependencies:

```txt
# COMET evaluation model for translation quality
comet-ml>=3.49.11

# COMET model for neural translation evaluation
# Note: This requires PyTorch. Install with: pip install comet-ml[model]
unbabel-comet>=2.2.2
```

### 2. Core Functionality Added (`src/evaluate_translation.py`)

#### New Imports

- Added `datetime` for timestamps
- Added `sacrebleu` for proper BLEU calculation
- Added conditional `comet` imports with graceful fallback

#### New Functions

- `calculate_sacrebleu_score()`: Proper BLEU calculation using sacrebleu
- `comet_score()`: Neural evaluation using COMET models
- `extract_source_text()`: Helper to extract source text from input format

#### Enhanced Dataset

- Added third test case: `{"input": "Spanish | Workday", "reference": "Workday"}`

#### Updated Main Function

- Streamlined evaluation process
- Added COMET scoring alongside BLEU
- Enhanced results structure with timestamps
- Dual quality gate checks (BLEU >= 45.0, COMET >= 0.3)
- Structured JSON output to `eval_results/latest.json`

### 3. Test Coverage (`tests/test_comet_evaluation.py`)

Added comprehensive tests for:

- Source text extraction
- BLEU score calculation with floating-point precision handling
- COMET score calculation (with mocking)
- Error handling for edge cases
- Results output and file handling
- Integration testing

## Usage

### Basic Usage

```python
from src.evaluate_translation import calculate_sacrebleu_score, comet_score

# Calculate BLEU score
hypotheses = ["nube nómina", "Workday"]
references = ["nube nómina", "Workday"]
bleu_score = calculate_sacrebleu_score(hypotheses, references)

# Calculate COMET score (requires model download on first run)
sources = ["cloud payroll", "Workday"]
comet_score_value = comet_score(references, hypotheses, sources)
```

### Running Evaluation

```bash
# Run the evaluation script
python src/evaluate_translation.py

# Run tests
python -m pytest tests/test_comet_evaluation.py -v
```

## Output Format

The evaluation now produces structured results saved to `eval_results/latest.json`:

```json
{
  "timestamp": "2025-07-08T09:18:33.123456",
  "BLEU": 85.5,
  "COMET": 0.75,
  "hypotheses": ["translation 1", "translation 2"],
  "references": ["reference 1", "reference 2"],
  "sources": ["source 1", "source 2"],
  "test_cases": 2
}
```

## Quality Gates

The evaluation includes quality gate checks for CI/CD:

- **BLEU threshold**: 45.0
- **COMET threshold**: 0.3

If either threshold is not met, the script exits with code 1, failing the CI/CD pipeline.

## Notes

- **First-time setup**: COMET model download may take several minutes on first run
- **Graceful fallback**: If COMET is not available, evaluation continues with BLEU only
- **Windows compatibility**: All file operations handle Unicode encoding properly
- **Floating-point precision**: Tests handle minor floating-point precision differences

## Implementation Details

### COMET Integration

- Uses `Unbabel/wmt22-comet-da` model
- Handles both tuple and dict return formats from different COMET versions
- Includes proper error handling and logging
- Configurable batch size and GPU usage (defaulted to CPU)

### BLEU Calculation

- Uses sentence-level BLEU averaged across the dataset
- More appropriate for small evaluation sets
- Handles empty inputs and mismatched lengths gracefully

### Error Handling

- Comprehensive exception handling throughout
- Detailed logging for debugging
- Graceful degradation when dependencies are missing

This implementation follows the repository's coding guidelines with clarity-first approach, comprehensive logging, and full test coverage.
