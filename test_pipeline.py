"""
Tests for the FinBERT Sentiment Pipeline

These are basic structural tests that verify the code organization
without requiring the actual model dependencies to be installed.
"""

import sys
import ast


def test_finbert_pipeline_structure():
    """Test that finbert_pipeline.py has the expected structure."""
    with open('finbert_pipeline.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Check for FinBERTSentimentPipeline class
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert 'FinBERTSentimentPipeline' in classes, "FinBERTSentimentPipeline class not found"
    
    # Check for expected methods
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'FinBERTSentimentPipeline':
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
    
    expected_methods = ['__init__', 'analyze_sentiment', '_analyze_single', 'batch_analyze']
    for method in expected_methods:
        assert method in methods, f"Method {method} not found in FinBERTSentimentPipeline"
    
    print("✓ finbert_pipeline.py structure is correct")
    return True


def test_example_structure():
    """Test that example.py has the expected structure."""
    with open('example.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Check for main function
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert 'main' in functions, "main function not found in example.py"
    
    print("✓ example.py structure is correct")
    return True


def test_imports():
    """Test that import statements are valid."""
    # Test finbert_pipeline.py imports
    with open('finbert_pipeline.py', 'r') as f:
        code = f.read()
        assert 'import torch' in code or 'from torch' in code
        assert 'from transformers import' in code
        assert 'import numpy' in code or 'from numpy' in code
    
    # Test example.py imports
    with open('example.py', 'r') as f:
        code = f.read()
        assert 'from finbert_pipeline import FinBERTSentimentPipeline' in code
    
    print("✓ Import statements are correct")
    return True


def test_requirements():
    """Test that requirements.txt has the necessary dependencies."""
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    assert 'torch' in requirements, "torch not in requirements.txt"
    assert 'transformers' in requirements, "transformers not in requirements.txt"
    assert 'numpy' in requirements, "numpy not in requirements.txt"
    
    print("✓ requirements.txt has all necessary dependencies")
    return True


def test_readme():
    """Test that README.md has expected sections."""
    with open('README.md', 'r') as f:
        readme = f.read()
    
    expected_sections = [
        '# FinBERT Sentiment Pipeline',
        '## Overview',
        '## Installation',
        '## Usage',
        '## Requirements'
    ]
    
    for section in expected_sections:
        assert section in readme, f"Section '{section}' not found in README.md"
    
    print("✓ README.md has all expected sections")
    return True


if __name__ == '__main__':
    print("Running FinBERT Sentiment Pipeline Tests")
    print("=" * 60)
    
    tests = [
        test_finbert_pipeline_structure,
        test_example_structure,
        test_imports,
        test_requirements,
        test_readme
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if failed > 0:
        print(f"Tests failed: {failed}/{len(tests)}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
