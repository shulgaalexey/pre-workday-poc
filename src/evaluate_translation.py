"""
Translation evaluation module with BLEU and COMET score calculation.

Evaluates translation quality using:
- BLEU scores via sacrebleu library
- COMET neural evaluation metrics
- LangChain evaluators for exact matching

Built for Windows + VS Code environment with clarity-first approach.
Includes quality gate checks for CI/CD pipelines.
"""

import datetime
import json
import logging
import os
import pathlib
import re
import sys
from typing import Any, Dict, List

import sacrebleu
from dotenv import load_dotenv
from langchain.evaluation import load_evaluator

try:
    from .agent import create_langchain_agent
    from .openai_config import check_openai_api_key, log_api_key_debug_info
except ImportError:
    # Fallback for direct script execution
    from agent import create_langchain_agent
    from openai_config import check_openai_api_key, log_api_key_debug_info

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import COMET for neural evaluation
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
    logger.info("COMET is available for neural evaluation")
except ImportError:
    COMET_AVAILABLE = False
    logger.warning("COMET not available. Install with: pip install unbabel-comet")

# Tiny test set
DATASET = [
    {"input": "Spanish | cloud payroll", "reference": "nube nómina"},
    {"input": "German | Workday payroll", "reference": "Workday Lohnabrechnung"},
    {"input": "Spanish | Workday", "reference": "Workday"}
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


def calculate_sacrebleu_score(hypotheses: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score using sacrebleu library.

    Args:
        hypotheses: List of predicted translations
        references: List of reference translations

    Returns:
        BLEU score as float (0-100)
    """
    try:
        if not hypotheses or not references:
            logger.warning("Empty hypotheses or references for BLEU calculation")
            return 0.0

        if len(hypotheses) != len(references):
            logger.warning("Mismatched lengths for BLEU calculation")
            return 0.0

        # For small datasets, calculate average sentence-level BLEU
        # This is more appropriate for PoC evaluation
        total_score = 0.0
        valid_scores = 0

        for hyp, ref in zip(hypotheses, references):
            if hyp and ref:
                sentence_bleu = sacrebleu.sentence_bleu(hyp, [ref])
                total_score += sentence_bleu.score
                valid_scores += 1

        if valid_scores == 0:
            logger.warning("No valid sentence pairs for BLEU calculation")
            return 0.0

        avg_score = total_score / valid_scores
        logger.info(f"SacreBLEU score: {avg_score:.2f} (averaged over {valid_scores} sentences)")
        return avg_score

    except Exception as e:
        logger.error(f"Error calculating SacreBLEU score: {e}")
        return 0.0


def comet_score(refs: List[str], hyps: List[str], srcs: List[str]) -> float:
    """
    Calculate COMET score using neural evaluation model.

    Args:
        refs: List of reference translations
        hyps: List of hypothesis/predicted translations
        srcs: List of source texts

    Returns:
        COMET score as float (typically 0-1, higher is better)
    """
    if not COMET_AVAILABLE:
        logger.warning("COMET not available, returning 0.0")
        return 0.0

    try:
        if not refs or not hyps or not srcs:
            logger.warning("Empty inputs for COMET calculation")
            return 0.0

        if len(refs) != len(hyps) or len(refs) != len(srcs):
            logger.error("Mismatched lengths for COMET calculation")
            return 0.0

        logger.info("Downloading COMET model (this may take a while on first run)")
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)

        # Prepare data for COMET
        data = []
        for src, mt, ref in zip(srcs, hyps, refs):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })

        logger.info(f"Calculating COMET score for {len(data)} examples")
        model_output = model.predict(data, batch_size=8, gpus=0)

        # COMET returns (seg_scores, sys_score)
        if isinstance(model_output, tuple) and len(model_output) == 2:
            seg_scores, sys_score = model_output
            logger.info(f"COMET system score: {sys_score:.4f}")
            return sys_score
        else:
            # Handle different COMET versions
            sys_score = model_output.get('system_score', 0.0)
            logger.info(f"COMET system score: {sys_score:.4f}")
            return sys_score

    except Exception as e:
        logger.error(f"Error calculating COMET score: {e}")
        return 0.0


def extract_source_text(input_text: str) -> str:
    """
    Extract source text from input format "Language | text".

    Args:
        input_text: Input in format "Language | text"

    Returns:
        Extracted source text
    """
    if "|" in input_text:
        return input_text.split("|", 1)[1].strip()
    return input_text.strip()


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
                # Get agent response - include chat_history for conversational agents
                result = agent.invoke({
                    "input": test_case["input"],
                    "chat_history": []
                })
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
    Main function to run translation evaluation with BLEU and COMET scores.

    Exits with code 1 if scores are below threshold (for CI/CD).
    """
    # Load environment variables from .env file
    load_dotenv()

    try:
        # Debug environment variables using centralized function
        logger.info("Starting translation evaluation")
        log_api_key_debug_info()

        # Check if OPENAI_API_KEY is available using centralized function
        openai_key = check_openai_api_key()
        if not openai_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            logger.error("Ensure OPENAI_API_KEY is set as a GitHub secret or in your .env file")
            print("[ERROR] OPENAI_API_KEY not found in environment variables")
            sys.exit(1)

        # Create agent
        agent = create_langchain_agent()
        logger.info("Agent created successfully")

        # Generate translations for all test cases
        hypotheses = []
        references = []
        sources = []

        for i, test_case in enumerate(DATASET):
            try:
                # Get agent response
                result = agent.invoke({
                    "input": test_case["input"],
                    "chat_history": []
                })
                raw_prediction = result.get("output", "")

                # Extract just the translation from the agent response
                prediction = extract_translation_from_response(raw_prediction)

                hypotheses.append(prediction)
                references.append(test_case["reference"])
                sources.append(extract_source_text(test_case["input"]))

                logger.info(f"Test case {i+1}: '{prediction}' vs '{test_case['reference']}'")

            except Exception as e:
                logger.error(f"Error processing test case {i}: {e}")
                hypotheses.append("")
                references.append(test_case["reference"])
                sources.append(extract_source_text(test_case["input"]))

        # Calculate BLEU score using sacrebleu
        bleu_score = calculate_sacrebleu_score(hypotheses, references)

        # Calculate COMET score
        comet_score_value = comet_score(references, hypotheses, sources)

        # Create results object
        results = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "BLEU": bleu_score,
            "COMET": comet_score_value,
            "hypotheses": hypotheses,
            "references": references,
            "sources": sources,
            "test_cases": len(DATASET)
        }

        # Save results to file
        pathlib.Path("eval_results").mkdir(exist_ok=True)
        output_file = pathlib.Path("eval_results/latest.json")

        # Ensure safe JSON serialization for Windows
        try:
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            output_file.write_text(results_json, encoding='utf-8')
        except UnicodeEncodeError as e:
            logger.warning(f"Unicode encoding issue in results: {e}")
            # Fallback to ASCII-safe output
            safe_results = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in results.items()}
            results_json = json.dumps(safe_results, indent=2, ensure_ascii=True)
            output_file.write_text(results_json, encoding='utf-8')

        # Print results
        print(f"BLEU: {bleu_score:.2f}")
        print(f"COMET: {comet_score_value:.4f}")
        print(f"Results saved to: {output_file}")

        # Set thresholds
        BLEU_THRESHOLD = 45.0
        COMET_THRESHOLD = 0.3

        # Check quality gates
        bleu_pass = bleu_score >= BLEU_THRESHOLD
        comet_pass = comet_score_value >= COMET_THRESHOLD

        if bleu_pass and comet_pass:
            print(f"[SUCCESS] Quality gates passed - BLEU: {bleu_score:.2f} >= {BLEU_THRESHOLD}, COMET: {comet_score_value:.4f} >= {COMET_THRESHOLD}")
            sys.exit(0)
        else:
            failure_reasons = []
            if not bleu_pass:
                failure_reasons.append(f"BLEU: {bleu_score:.2f} < {BLEU_THRESHOLD}")
            if not comet_pass:
                failure_reasons.append(f"COMET: {comet_score_value:.4f} < {COMET_THRESHOLD}")

            print(f"[FAIL] Quality gate failed - {', '.join(failure_reasons)}")
            raise SystemExit("Quality gate failed")

    except SystemExit:
        # Re-raise SystemExit to preserve exit code
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"[ERROR] Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# COMET and BLEU evaluation implementation
# This module now includes:
# - COMET neural evaluation metrics via unbabel-comet
# - Proper BLEU calculation via sacrebleu
# - Quality gate checks for CI/CD
# - Structured results output with timestamps
# - Comprehensive test dataset evaluation
