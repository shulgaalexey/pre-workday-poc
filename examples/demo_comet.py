"""
Demo script to test the COMET evaluation functionality.

This script runs a simplified version of the evaluation without requiring
a real OpenAI API key, using mocked agent responses.
"""

import os
# Fix OpenMP library conflict issue in CI/CD environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluate_translation import (
    calculate_sacrebleu_score,
    comet_score,
    extract_source_text,
    DATASET,
    COMET_AVAILABLE
)

def demo_comet_evaluation():
    """Demo the COMET evaluation functionality."""
    print("=== COMET Evaluation Demo ===\n")

    # Test data
    hypotheses = ["nube nómina", "Workday Lohnabrechnung", "Workday"]
    references = ["nube nómina", "Workday Lohnabrechnung", "Workday"]
    sources = ["cloud payroll", "Workday payroll", "Workday"]

    print("Test data:")
    for i, (hyp, ref, src) in enumerate(zip(hypotheses, references, sources)):
        print(f"  {i+1}. Source: '{src}' | Hypothesis: '{hyp}' | Reference: '{ref}'")
    print()

    # Test BLEU calculation
    print("1. Testing BLEU calculation:")
    bleu_score = calculate_sacrebleu_score(hypotheses, references)
    print(f"   BLEU score: {bleu_score:.2f}")
    print()

    # Test source text extraction
    print("2. Testing source text extraction:")
    for item in DATASET:
        extracted = extract_source_text(item["input"])
        print(f"   '{item['input']}' -> '{extracted}'")
    print()

    # Test COMET calculation
    print("3. Testing COMET calculation:")
    if COMET_AVAILABLE:
        print("   COMET is available!")
        # This would take a long time on first run due to model download
        print("   Skipping actual COMET calculation to avoid long download time...")
        print("   COMET would return a score between 0 and 1 (higher is better)")
    else:
        print("   COMET is not available. Install with: pip install unbabel-comet")
    print()

    # Test with mock COMET for demonstration
    print("4. Testing COMET calculation (mocked):")
    with patch('src.evaluate_translation.COMET_AVAILABLE', True), \
         patch('src.evaluate_translation.download_model') as mock_download, \
         patch('src.evaluate_translation.load_from_checkpoint') as mock_load:

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = ([], 0.85)  # seg_scores, sys_score
        mock_load.return_value = mock_model
        mock_download.return_value = "mocked/model/path"

        comet_score_value = comet_score(references, hypotheses, sources)
        print(f"   Mocked COMET score: {comet_score_value:.4f}")
    print()

    print("=== Demo completed successfully! ===")

if __name__ == "__main__":
    demo_comet_evaluation()
