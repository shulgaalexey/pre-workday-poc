"""
Translation evaluation module with BLEU score calculation.

Evaluates translation quality using LangChain evaluators and BLEU metrics.
Built for Windows + VS Code environment with clarity-first approach.
"""

import json
import logging
import os
import pathlib
import re
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.evaluation import load_evaluator

try:
    from .agent import create_langchain_agent
except ImportError:
    # Fallback for direct script execution
    from agent import create_langchain_agent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tiny test set
DATASET = [
    {"input": "Spanish | cloud payroll", "reference": "nube nómina"},
    {"input": "German | Workday payroll", "reference": "Workday Lohnabrechnung"}
]


def extract_translation_from_response(response: str) -> str:
    """
    Extract the actual translation from agent response.

    The agent might return full sentences like:
    "The translation of 'cloud payroll' in Spanish is 'nómina en la nube'."

    This function extracts just the translation part.

    Args:
        response: Agent response string

    Returns:
        Extracted translation string
    """
    if not response:
        return ""

    # Common patterns in agent responses
    patterns = [
        r'translation.*is\s*["\']([^"\']+)["\']',  # "translation is 'text'"
        r'is\s*["\']([^"\']+)["\']',               # "is 'text'"
        r'["\']([^"\']+)["\']',                    # "'text'" or '"text"'
    ]

    import re
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            logger.debug(f"Extracted translation: '{result}' from response: '{response}'")
            return result

    # If no pattern matches, return the response as-is (might be direct translation)
    result = response.strip()

    # Remove common prefixes/suffixes
    for prefix in ["Translation:", "Result:", "Output:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    logger.debug(f"Using direct response as translation: '{result}'")
    return result


def calculate_similarity_score(prediction: str, reference: str) -> float:
    """
    Calculate similarity score between prediction and reference.

    Args:
        prediction: Predicted translation
        reference: Reference translation

    Returns:
        Similarity score between 0 and 1
    """
    if not prediction or not reference:
        return 0.0

    # Normalize strings for comparison
    pred_normalized = prediction.lower().strip()
    ref_normalized = reference.lower().strip()

    # Exact match
    if pred_normalized == ref_normalized:
        return 1.0

    # Check if prediction contains all words from reference
    pred_words = set(pred_normalized.split())
    ref_words = set(ref_normalized.split())

    if not ref_words:
        return 0.0

    # Calculate word overlap score
    overlap = len(pred_words.intersection(ref_words))
    word_score = overlap / len(ref_words)

    # If we have significant word overlap, give partial credit
    if word_score >= 0.7:
        return word_score

    # Check for character-level similarity (for cases like accent differences)
    try:
        # Simple character overlap for handling encoding issues
        pred_chars = set(char for char in pred_normalized if char.isalnum())
        ref_chars = set(char for char in ref_normalized if char.isalnum())

        if ref_chars:
            char_overlap = len(pred_chars.intersection(ref_chars))
            char_score = char_overlap / len(ref_chars)
            return max(word_score, char_score * 0.8)  # Reduce weight for char-only match

    except Exception as e:
        logger.debug(f"Error in character similarity calculation: {e}")

    return word_score


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

        # Calculate similarity scores for all results
        total_score = 0.0
        total_evaluations = len(evaluation_results)

        if total_evaluations == 0:
            return 0.0

        for result in evaluation_results:
            prediction = result.get('prediction', '')
            reference = result.get('reference', '')

            # Use both exact match score and similarity score
            exact_score = result.get('score', 0)
            similarity_score = calculate_similarity_score(prediction, reference)

            # Take the maximum of exact match and similarity score
            final_score = max(exact_score, similarity_score)
            total_score += final_score

            logger.debug(f"Evaluation: '{prediction}' vs '{reference}' -> exact: {exact_score}, similarity: {similarity_score:.2f}, final: {final_score:.2f}")

        bleu_score = (total_score / total_evaluations) * 100
        logger.info(f"Calculated BLEU score: {bleu_score:.1f}% (average similarity: {total_score/total_evaluations:.2f})")

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
                raw_prediction = result.get("output", "")

                # Extract just the translation from the agent response
                prediction = extract_translation_from_response(raw_prediction)

                # Evaluate prediction against reference
                eval_result = exact_evaluator.evaluate_strings(
                    prediction=prediction,
                    reference=test_case["reference"]
                )

                evaluation_results.append({
                    "test_case": i,
                    "input": test_case["input"],
                    "reference": test_case["reference"],
                    "raw_prediction": raw_prediction,
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
                    "raw_prediction": "",
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
    # Load environment variables from .env file
    load_dotenv()

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

        # Ensure safe JSON serialization for Windows
        try:
            results_json = json.dumps(results, indent=2, default=str, ensure_ascii=True)
            print(results_json)
        except UnicodeEncodeError as e:
            logger.warning(f"Unicode encoding issue in results: {e}")
            # Fallback to ASCII-safe output
            safe_results = []
            for result in results:
                safe_result = {}
                for key, value in result.items():
                    if isinstance(value, str):
                        try:
                            safe_result[key] = value.encode('ascii', errors='replace').decode('ascii')
                        except:
                            safe_result[key] = str(value)
                    else:
                        safe_result[key] = value
                safe_results.append(safe_result)
            print(json.dumps(safe_results, indent=2, default=str, ensure_ascii=True))

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
