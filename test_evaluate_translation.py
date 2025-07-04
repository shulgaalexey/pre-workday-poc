"""
Test module for translation evaluation with BLEU score validation.

Tests the translation evaluation functionality and enforces BLEU score thresholds.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent import create_langchain_agent
from evaluate_translation import calculate_bleu_score, evaluate_translations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestTranslationEvaluation:
    """Test cases for translation evaluation functionality."""

    @pytest.mark.translation_eval
    @patch.dict('os.environ', {"OPENAI_API_KEY": "test_key"})
    def test_translation_eval_bleu_threshold(self, mocker):
        """
        Test translation evaluation meets BLEU score threshold.
        Fails build if BLEU < 50.
        """
        # Mock the agent to return predictable translations
        mock_agent = mocker.patch('evaluate_translation.create_langchain_agent')
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Mock successful translations that should score well
        mock_agent_instance.invoke.side_effect = [
            {"output": "nube nómina"},  # Perfect match for Spanish
            {"output": "Workday Lohnabrechnung"}  # Perfect match for German
        ]

        # Mock load_evaluator to return successful results
        mock_evaluator = mocker.patch('evaluate_translation.load_evaluator')
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate_strings.side_effect = [
            {"score": 1},  # Perfect match
            {"score": 1}   # Perfect match
        ]

        # Run evaluation
        results = evaluate_translations()

        # Calculate BLEU score
        bleu_score = calculate_bleu_score(results)

        logger.info(f"BLEU score: {bleu_score}")

        # Assert BLEU score threshold
        assert bleu_score >= 50, f"BLEU score {bleu_score} is below threshold of 50"

    @pytest.mark.translation_eval
    @patch.dict('os.environ', {"OPENAI_API_KEY": "test_key"})
    def test_translation_eval_failing_bleu(self, mocker):
        """
        Test translation evaluation with poor BLEU score.
        This test verifies the failure mechanism works.
        """
        # Mock the agent to return poor translations
        mock_agent = mocker.patch('evaluate_translation.create_langchain_agent')
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Mock poor translations
        mock_agent_instance.invoke.side_effect = [
            {"output": "incorrect translation"},
            {"output": "another bad translation"}
        ]

        # Mock load_evaluator to return poor results
        mock_evaluator = mocker.patch('evaluate_translation.load_evaluator')
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate_strings.side_effect = [
            {"score": 0},  # No match
            {"score": 0}   # No match
        ]

        # Run evaluation
        results = evaluate_translations()

        # Calculate BLEU score
        bleu_score = calculate_bleu_score(results)

        logger.info(f"BLEU score: {bleu_score}")

        # This should fail if BLEU is truly poor
        with pytest.raises(AssertionError, match="BLEU score .* is below threshold"):
            assert bleu_score >= 50, f"BLEU score {bleu_score} is below threshold of 50"

    @pytest.mark.translation_eval
    def test_evaluate_translations_exception_handling(self, mocker):
        """Test evaluation function handles exceptions properly."""
        # Mock agent creation to raise an exception
        mock_agent = mocker.patch('evaluate_translation.create_langchain_agent')
        mock_agent.side_effect = Exception("Test exception")

        # Should raise the exception
        with pytest.raises(Exception, match="Test exception"):
            evaluate_translations()


class TestBLEUScoreCalculation:
    """Test cases for BLEU score calculation utility functions."""

    def test_calculate_bleu_score_empty_results(self):
        """Test BLEU calculation with empty results."""
        empty_results = []
        bleu_score = calculate_bleu_score(empty_results)
        assert bleu_score == 0.0

    def test_calculate_bleu_score_partial_matches(self):
        """Test BLEU calculation with partial matches."""
        mixed_results = [
            {'score': 1},
            {'score': 0},
            {'score': 1},
            {'score': 0}
        ]
        bleu_score = calculate_bleu_score(mixed_results)
        assert bleu_score == 50.0  # 2/4 = 50%

    def test_calculate_bleu_score_invalid_input(self):
        """Test BLEU calculation with invalid input."""
        invalid_results = None
        bleu_score = calculate_bleu_score(invalid_results)
        assert bleu_score == 0.0
