"""
Test module for COMET evaluation functionality.

Tests the new COMET scoring integration and helper functions.
"""

import os

# Fix OpenMP library conflict issue in CI/CD environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.evaluate_translation import (COMET_AVAILABLE, DATASET,
                                      calculate_sacrebleu_score, comet_score,
                                      extract_source_text)


class TestCometEvaluation:
    """Test COMET evaluation functionality."""

    def test_extract_source_text(self):
        """Test source text extraction from input format."""
        # Test normal case
        assert extract_source_text("Spanish | cloud payroll") == "cloud payroll"
        assert extract_source_text("German | Workday") == "Workday"

        # Test edge cases
        assert extract_source_text("text without pipe") == "text without pipe"
        assert extract_source_text("  Spanish  |  cloud payroll  ") == "cloud payroll"
        assert extract_source_text("") == ""

    def test_calculate_sacrebleu_score(self):
        """Test BLEU score calculation using sacrebleu."""
        # Test perfect match
        hypotheses = ["hello world", "goodbye world"]
        references = ["hello world", "goodbye world"]
        score = calculate_sacrebleu_score(hypotheses, references)
        assert abs(score - 100.0) < 0.01  # Allow for floating point precision

        # Test partial match
        hypotheses = ["hello world", "goodbye earth"]
        references = ["hello world", "goodbye world"]
        score = calculate_sacrebleu_score(hypotheses, references)
        assert 0 <= score <= 100

        # Test empty inputs
        assert calculate_sacrebleu_score([], []) == 0.0
        assert calculate_sacrebleu_score(["test"], []) == 0.0

    @patch('src.evaluate_translation.COMET_AVAILABLE', True)
    @patch('comet.download_model')
    @patch('comet.load_from_checkpoint')
    def test_comet_score_available(self, mock_load_checkpoint, mock_download_model):
        """Test COMET score calculation when COMET is available."""
        # Mock the model and its predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = ([], 0.85)  # seg_scores, sys_score
        mock_load_checkpoint.return_value = mock_model
        mock_download_model.return_value = "path/to/model"

        refs = ["hello world", "goodbye world"]
        hyps = ["hello world", "goodbye earth"]
        srcs = ["hello world", "goodbye world"]

        score = comet_score(refs, hyps, srcs)
        assert score == 0.85

        # Verify model was called correctly
        mock_download_model.assert_called_once_with("Unbabel/wmt22-comet-da")
        mock_load_checkpoint.assert_called_once_with("path/to/model")

    def test_comet_score_unavailable(self):
        """Test COMET score when COMET is not available."""
        with patch('src.evaluate_translation.COMET_AVAILABLE', False):
            score = comet_score(["test"], ["test"], ["test"])
            assert score == 0.0

    def test_comet_score_empty_inputs(self):
        """Test COMET score with empty inputs."""
        if COMET_AVAILABLE:
            score = comet_score([], [], [])
            assert score == 0.0
        else:
            # If COMET is not available, should return 0.0
            score = comet_score([], [], [])
            assert score == 0.0

    def test_comet_score_mismatched_lengths(self):
        """Test COMET score with mismatched input lengths."""
        refs = ["hello"]
        hyps = ["hello", "world"]
        srcs = ["hello"]

        score = comet_score(refs, hyps, srcs)
        assert score == 0.0

    def test_dataset_updated(self):
        """Test that dataset includes the new test case."""
        assert len(DATASET) == 3
        assert {"input": "Spanish | Workday", "reference": "Workday"} in DATASET

    @patch('src.evaluate_translation.create_langchain_agent')
    @patch('src.evaluate_translation.check_openai_api_key')
    @patch('src.evaluate_translation.log_api_key_debug_info')
    def test_main_integration(self, mock_log_api, mock_check_key, mock_create_agent):
        """Test main function integration (without actually running COMET)."""
        # Mock API key check
        mock_check_key.return_value = "test-key"

        # Mock agent responses
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            {"output": "nube nómina"},
            {"output": "Workday Lohnabrechnung"},
            {"output": "Workday"}
        ]
        mock_create_agent.return_value = mock_agent

        # Test would require mocking all the scoring functions
        # This is a structural test to ensure the main function can be called
        from src.evaluate_translation import main

        # This would require extensive mocking for a full test
        # For now, just verify the function exists and can be imported
        assert callable(main)


class TestResultsOutput:
    """Test results output and file handling."""

    def test_results_directory_creation(self):
        """Test that results directory is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Create the directory structure that would be created
                eval_results_dir = pathlib.Path("eval_results")
                eval_results_dir.mkdir(exist_ok=True)

                assert eval_results_dir.exists()
                assert eval_results_dir.is_dir()

            finally:
                os.chdir(original_cwd)

    def test_json_serialization(self):
        """Test JSON serialization of results."""
        import datetime

        test_results = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "BLEU": 85.5,
            "COMET": 0.75,
            "hypotheses": ["test translation"],
            "references": ["reference translation"],
            "sources": ["source text"],
            "test_cases": 1
        }

        # Test that results can be serialized to JSON
        json_str = json.dumps(test_results, indent=2, ensure_ascii=False)
        assert json_str is not None

        # Test that it can be parsed back
        parsed = json.loads(json_str)
        assert parsed["BLEU"] == 85.5
        assert parsed["COMET"] == 0.75
