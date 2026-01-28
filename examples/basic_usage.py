"""
Basic usage example for the FinBERT Sentiment Pipeline.

This script demonstrates how to use the FinBERTSentimentPipeline class
to analyze sentiment in financial texts.
"""

import sys
import os

# Add parent directory to path to import finbert_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finbert_pipeline import FinBERTSentimentPipeline


def main():
    """Demonstrate basic usage of the FinBERT sentiment pipeline."""
    
    # Initialize the pipeline
    print("Initializing FinBERT Sentiment Pipeline...")
    pipeline = FinBERTSentimentPipeline()
    print("Pipeline initialized successfully!\n")
    
    # Example 1: Analyze a single text
    print("=" * 60)
    print("Example 1: Single Text Analysis")
    print("=" * 60)
    
    text = "The company reported strong earnings growth in Q4."
    print(f"\nAnalyzing: '{text}'")
    result = pipeline.analyze(text)
    
    print(f"\nResults:")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Detailed Scores:")
    for label, score in result['scores'].items():
        print(f"    {label.capitalize()}: {score:.2%}")
    
    # Example 2: Analyze multiple texts
    print("\n" + "=" * 60)
    print("Example 2: Batch Text Analysis")
    print("=" * 60)
    
    texts = [
        "Shares plummeted after disappointing revenue results.",
        "The market remained stable throughout the trading session.",
        "Investors are optimistic about the upcoming merger.",
        "The CEO resigned amid financial irregularities."
    ]
    
    results = pipeline.analyze_batch(texts)
    
    print(f"\nAnalyzing {len(texts)} texts:\n")
    for i, (text, result) in enumerate(zip(texts, results), 1):
        if 'error' in result:
            print(f"{i}. Text: '{text}'")
            print(f"   Error: {result['error']}")
            if 'text_index' in result:
                print(f"   (Error at index {result['text_index']})")
            print()
        else:
            print(f"{i}. Text: '{text}'")
            print(f"   Sentiment: {result['sentiment']} ({result['confidence']:.2%})\n")
    
    # Example 3: Custom text from user
    print("=" * 60)
    print("Example 3: Analyze Your Own Text")
    print("=" * 60)
    
    custom_text = input("\nEnter a financial text to analyze (or press Enter to skip): ")
    
    if custom_text.strip():
        result = pipeline.analyze(custom_text)
        print(f"\nResults for your text:")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Detailed Scores:")
        for label, score in result['scores'].items():
            print(f"    {label.capitalize()}: {score:.2%}")
    else:
        print("\nSkipping custom text analysis.")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
