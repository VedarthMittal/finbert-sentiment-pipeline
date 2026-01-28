# FinBERT Sentiment Pipeline

A sentiment analysis pipeline using FinBERT (Financial BERT) for analyzing financial text and news.

## Overview

This project implements a sentiment analysis pipeline specifically designed for financial text using the FinBERT model. FinBERT is a pre-trained NLP model fine-tuned on financial communication text, making it ideal for analyzing sentiment in financial news, reports, and other finance-related documents.

## Features

- Sentiment analysis for financial text
- Pre-trained FinBERT model integration
- Easy-to-use pipeline interface
- Support for batch processing

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/VedarthMittal/finbert-sentiment-pipeline.git
cd finbert-sentiment-pipeline
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Git Setup Instructions

If you're setting up this repository for the first time from an **existing local directory** (not cloning from GitHub), follow these steps to link your local repository to GitHub:

### On Windows:
```cmd
cd "path\to\your\project\directory"
git remote add origin https://github.com/[USERNAME]/finbert-sentiment-pipeline.git
git branch -M main
git push -u origin main
```

### On macOS/Linux:
```bash
cd /path/to/your/project/directory
git remote add origin https://github.com/[USERNAME]/finbert-sentiment-pipeline.git
git branch -M main
git push -u origin main
```

**Note:** 
- Replace `[USERNAME]` with your actual GitHub username.
- Replace `path\to\your\project\directory` or `/path/to/your/project/directory` with your actual project path.
- These commands are for linking an existing local repository. If you're starting fresh, use the clone command shown in the Installation section instead.

## Usage

```python
from finbert_pipeline import FinBERTSentimentPipeline

# Initialize the pipeline
pipeline = FinBERTSentimentPipeline()

# Analyze sentiment
text = "The company reported strong earnings growth in Q4."
result = pipeline.analyze(text)
print(result)
```

## Project Structure

```
finbert-sentiment-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── finbert_pipeline.py
└── examples/
    └── basic_usage.py
```

## Dependencies

- transformers
- torch
- numpy
- pandas
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- FinBERT model by ProsusAI
- Hugging Face Transformers library

## Contact

For questions or feedback, please open an issue on GitHub.