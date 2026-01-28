"""
Example usage of the FinBERT Sentiment Pipeline

This script demonstrates various ways to use the FinBERT sentiment analysis pipeline.
"""

from finbert_pipeline import FinBERTSentimentPipeline


def main():
    print("="*80)
    print("FinBERT Sentiment Analysis Pipeline - Examples")
    print("="*80)
    print()
    
    # Initialize the pipeline
    print("Initializing FinBERT pipeline...")
    pipeline = FinBERTSentimentPipeline()
    print("Pipeline initialized successfully!")
    print()
    
    # Example 1: Single text analysis
    print("-" * 80)
    print("Example 1: Single Text Analysis")
    print("-" * 80)
    
    text = "The company's revenue increased by 20% this quarter, exceeding analyst expectations."
    result = pipeline.analyze_sentiment(text)
    
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Detailed Scores:")
    for label, score in result['scores'].items():
        print(f"  {label.capitalize()}: {score:.2%}")
    print()
    
    # Example 2: Multiple texts analysis
    print("-" * 80)
    print("Example 2: Multiple Texts Analysis")
    print("-" * 80)
    
    financial_texts = [
        "The stock price fell sharply after the disappointing earnings report.",
        "Investors are optimistic about the company's future growth prospects.",
        "The market remained relatively stable with no significant changes.",
        "Strong quarterly results led to a surge in share prices.",
        "The company faces challenges due to increased competition.",
        "Analysts predict steady performance in the coming quarters."
    ]
    
    results = pipeline.analyze_sentiment(financial_texts)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Sentiment: {result['sentiment'].upper()} (confidence: {result['confidence']:.2%})")
    
    print()
    
    # Example 3: Batch processing
    print("-" * 80)
    print("Example 3: Batch Processing (for large datasets)")
    print("-" * 80)
    
    large_dataset = [
        "Revenue growth exceeded expectations for the third consecutive quarter.",
        "The company announced layoffs affecting 10% of its workforce.",
        "Trading volume remained consistent with historical averages.",
        "New product launch drives investor enthusiasm.",
        "Regulatory concerns weigh on market sentiment.",
        "Dividend yield remains attractive for long-term investors.",
        "Profit margins compressed due to rising input costs.",
        "Strategic partnership announced with major industry player."
    ]
    
    batch_results = pipeline.batch_analyze(large_dataset, batch_size=4)
    
    print(f"Analyzed {len(batch_results)} texts in batches of 4")
    print("\nSentiment Distribution:")
    
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in batch_results:
        sentiment_counts[result['sentiment']] += 1
    
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(batch_results)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Example 4: High confidence predictions
    print("-" * 80)
    print("Example 4: Filtering High-Confidence Predictions")
    print("-" * 80)
    
    high_confidence_threshold = 0.90
    high_confidence_results = [r for r in batch_results if r['confidence'] >= high_confidence_threshold]
    
    print(f"\nPredictions with confidence >= {high_confidence_threshold:.0%}:")
    print(f"Found {len(high_confidence_results)} out of {len(batch_results)} results")
    
    for result in high_confidence_results:
        print(f"\n• {result['text'][:70]}...")
        print(f"  Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    
    print()
    print("="*80)
    print("Examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
