# FinBERT Sentiment Pipeline

A Python-based sentiment analysis pipeline using FinBERT, a pre-trained NLP model specifically designed for analyzing sentiment in financial text.

## Overview

FinBERT is a BERT-based model fine-tuned on financial text data. This pipeline provides an easy-to-use interface for analyzing sentiment (positive, negative, or neutral) in financial news, reports, and other financial documents.

## Features

- **Easy-to-use API**: Simple interface for sentiment analysis
- **Batch Processing**: Efficient batch analysis for multiple texts
- **Pre-trained Model**: Uses the ProsusAI/finbert model from Hugging Face
- **Confidence Scores**: Returns confidence scores for all sentiment categories

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VedarthMittal/finbert-sentiment-pipeline.git
cd finbert-sentiment-pipeline
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from finbert_pipeline import FinBERTSentimentPipeline

# Initialize the pipeline
pipeline = FinBERTSentimentPipeline()

# Analyze a single text
text = "The company's revenue increased by 20% this quarter."
result = pipeline.analyze_sentiment(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Scores: {result['scores']}")
```

### Analyzing Multiple Texts

```python
# Analyze multiple texts
texts = [
    "The stock price fell sharply after the earnings report.",
    "Investors are optimistic about the company's future prospects.",
    "The market remained stable with no significant changes."
]

results = pipeline.analyze_sentiment(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
```

### Batch Processing

For better performance with large datasets, use batch processing:

```python
# Batch process many texts
texts = [...]  # Your list of texts
results = pipeline.batch_analyze(texts, batch_size=16)
```

## Output Format

The pipeline returns a dictionary (or list of dictionaries) with the following structure:

```python
{
    "text": "The original input text",
    "sentiment": "positive",  # or "negative" or "neutral"
    "confidence": 0.95,       # Confidence score for the predicted sentiment
    "scores": {
        "positive": 0.95,
        "negative": 0.03,
        "neutral": 0.02
    }
}
```

## Example

Run the example script directly:

```bash
python finbert_pipeline.py
```

This will demonstrate basic usage with sample financial texts.

## Model Information

This pipeline uses the **ProsusAI/finbert** model from Hugging Face, which is a BERT-based model fine-tuned on financial sentiment analysis tasks.

- **Model**: ProsusAI/finbert
- **Base Model**: BERT
- **Training Data**: Financial news and reports
- **Sentiment Classes**: Positive, Negative, Neutral

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy 1.24+

See `requirements.txt` for complete dependencies.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- FinBERT model by ProsusAI
- Hugging Face Transformers library