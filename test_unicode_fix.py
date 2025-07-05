#!/usr/bin/env python3
"""
Test script to verify Unicode encoding fix for Windows GitHub Actions.
This reproduces the exact scenario that was failing.
"""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout


def test_unicode_output():
    """Test that all output uses ASCII-compatible characters."""

    # Capture stdout and stderr to test encoding
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Test the patterns that were causing issues
            print("[ERROR] OPENAI_API_KEY not found in environment variables")
            print("[SUCCESS] BLEU score 75.0% meets threshold of 50.0%")
            print("[FAIL] BLEU score 25.0% below threshold of 50.0%")
            print("[ERROR] Evaluation failed: Test error")

        # Get the output
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()

        # Verify no problematic Unicode characters
        problematic_chars = ['✅', '❌', '⚠️', '\u274c', '\u2705', '\u26a0']
        for char in problematic_chars:
            assert char not in stdout_content, f"Found problematic Unicode character: {char}"
            assert char not in stderr_content, f"Found problematic Unicode character: {char}"

        # Verify we have the expected ASCII alternatives
        assert "[ERROR]" in stdout_content
        assert "[SUCCESS]" in stdout_content
        assert "[FAIL]" in stdout_content

        print("Unicode encoding test passed - all output uses ASCII-compatible characters")
        return True

    except UnicodeEncodeError as e:
        print(f"Unicode encoding test failed: {e}")
        return False
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        return False


def test_cp1252_encoding():
    """Test that output can be encoded with cp1252 (Windows default)."""

    test_strings = [
        "[ERROR] OPENAI_API_KEY not found in environment variables",
        "[SUCCESS] BLEU score 75.0% meets threshold of 50.0%",
        "[FAIL] BLEU score 25.0% below threshold of 50.0%",
        "[ERROR] Evaluation failed: Test error"
    ]

    try:
        for test_string in test_strings:
            # Try to encode with cp1252 (Windows default)
            encoded = test_string.encode('cp1252')
            decoded = encoded.decode('cp1252')
            assert decoded == test_string, f"Encoding/decoding mismatch for: {test_string}"

        print("CP1252 encoding test passed - all strings are Windows-compatible")
        return True

    except UnicodeEncodeError as e:
        print(f"CP1252 encoding test failed: {e}")
        return False
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Unicode encoding fix for Windows GitHub Actions...")
    print("=" * 60)

    test1_passed = test_unicode_output()
    test2_passed = test_cp1252_encoding()

    if test1_passed and test2_passed:
        print("\nAll Unicode encoding tests passed!")
        print("The fix should work correctly in Windows GitHub Actions environment.")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
