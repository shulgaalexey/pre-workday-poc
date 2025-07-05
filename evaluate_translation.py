"""
Translation evaluation module with BLEU score calculation.

Evaluates translation quality using LangChain evaluators and BLEU metrics.
Built for Windows + VS Code environment with clarity-first approach.
"""

import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List

from langchain.evaluation import load_evaluator

from agent import create_langchain_agent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tiny test set
DATASET = [
    {"input": "Spanish | cloud payroll", "reference": "nube nómina"},
    {"input": "German | Workday payroll", "reference": "Workday Lohnabrechnung"}
]


def calculate_bleu_score(evaluation_results: List[Dict[str, Any]]) -> float:
    """
    Calculate BLEU score from evaluation results.

    Args:
        evaluation_results: List of evaluation results with scores

    Returns:
        BLEU score as percentage (0-100)

    Note:
        This is a simplified BLEU calculation for PoC purposes.
        In production, use proper BLEU libraries like sacrebleu.
    """
    try:
        if not evaluation_results:
            logger.warning("No evaluation results found")
            return 0.0

        # Simple exact match percentage as proxy for BLEU
        # In real implementation, use proper BLEU calculation
        exact_matches = sum(1 for result in evaluation_results if result.get('score') == 1)
        total_evaluations = len(evaluation_results)

        if total_evaluations == 0:
            return 0.0

        bleu_score = (exact_matches / total_evaluations) * 100
        logger.info(f"Calculated BLEU score: {bleu_score}% ({exact_matches}/{total_evaluations} exact matches)")

        return bleu_score

    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0


def evaluate_translations() -> List[Dict[str, Any]]:
    """
    Run translation evaluation on the test dataset.

    Returns:
        List of evaluation results
    """
    logger.info("Starting translation evaluation")

    try:
        # Create agent
        agent = create_langchain_agent()
        logger.info("Agent created successfully")

        # Create exact match evaluator
        exact_evaluator = load_evaluator("exact_match")

        # Run evaluation on each test case
        evaluation_results = []
        logger.info(f"Running evaluation on {len(DATASET)} test cases")

        for i, test_case in enumerate(DATASET):
            try:
                # Get agent response
                result = agent.invoke({"input": test_case["input"]})
                prediction = result.get("output", "")

                # Evaluate prediction against reference
                eval_result = exact_evaluator.evaluate_strings(
                    prediction=prediction,
                    reference=test_case["reference"]
                )

                evaluation_results.append({
                    "test_case": i,
                    "input": test_case["input"],
                    "reference": test_case["reference"],
                    "prediction": prediction,
                    "score": eval_result["score"]
                })

                logger.info(f"Test case {i+1}: '{prediction}' vs '{test_case['reference']}' -> {eval_result['score']}")

            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                evaluation_results.append({
                    "test_case": i,
                    "input": test_case["input"],
                    "reference": test_case["reference"],
                    "prediction": "",
                    "score": 0,
                    "error": str(e)
                })

        logger.info("Evaluation completed successfully")
        return evaluation_results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def main():
    """
    Main function to run translation evaluation and check BLEU threshold.

    Exits with code 1 if BLEU score is below threshold (for CI/CD).
    """
    try:
        # Debug environment variables
        logger.info("Starting translation evaluation")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info(f"OPENAI_API_KEY found (length: {len(openai_key)})")
        else:
            logger.error("OPENAI_API_KEY not found in environment variables")
            logger.error("Available environment variables:")
            for key in sorted(os.environ.keys()):
                if 'API' in key or 'OPENAI' in key:
                    logger.error(f"  {key}: {'SET' if os.environ[key] else 'EMPTY'}")

        # Check if OPENAI_API_KEY is available
        if not openai_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            logger.error("Ensure OPENAI_API_KEY is set as a GitHub secret or in your .env file")
            print("[ERROR] OPENAI_API_KEY not found in environment variables")
            sys.exit(1)

        # Run evaluation
        results = evaluate_translations()
        print("Evaluation Results:")
        print(json.dumps(results, indent=2, default=str))

        # Calculate and check BLEU score
        bleu_score = calculate_bleu_score(results)

        # Set threshold
        BLEU_THRESHOLD = 50.0

        if bleu_score >= BLEU_THRESHOLD:
            print(f"[SUCCESS] BLEU score {bleu_score}% meets threshold of {BLEU_THRESHOLD}%")
            sys.exit(0)
        else:
            print(f"[FAIL] BLEU score {bleu_score}% below threshold of {BLEU_THRESHOLD}%")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"[ERROR] Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# (Optional) COMET / BLEU via external libs here
# For production use, consider adding:
# - sacrebleu for proper BLEU calculation
# - COMET for neural evaluation metrics
# - More comprehensive test datasets
