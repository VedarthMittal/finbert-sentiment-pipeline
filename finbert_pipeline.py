"""
FinBERT Sentiment Pipeline

A sentiment analysis pipeline for financial text using the FinBERT model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class FinBERTSentimentPipeline:
    """
    A pipeline for sentiment analysis of financial text using FinBERT.
    
    The FinBERT model is specifically fine-tuned for financial sentiment analysis
    and classifies text into positive, negative, or neutral sentiment.
    """
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize the FinBERT sentiment pipeline.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
                             Default is "ProsusAI/finbert".
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.labels = ["positive", "negative", "neutral"]
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def analyze(self, text):
        """
        Analyze the sentiment of the given financial text.
        
        Args:
            text (str): The financial text to analyze.
            
        Returns:
            dict: A dictionary containing the sentiment label and confidence scores.
                  Example: {
                      'sentiment': 'positive',
                      'confidence': 0.95,
                      'scores': {
                          'positive': 0.95,
                          'negative': 0.02,
                          'neutral': 0.03
                      }
                  }
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convert to numpy for easier handling
        probs = probabilities[0].numpy()
        
        # Get the predicted sentiment
        predicted_class = np.argmax(probs)
        sentiment = self.labels[predicted_class]
        confidence = float(probs[predicted_class])
        
        # Create scores dictionary
        scores = {label: float(prob) for label, prob in zip(self.labels, probs)}
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): A list of financial texts to analyze.
            
        Returns:
            list: A list of dictionaries containing sentiment analysis results.
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of texts")
        
        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results


def main():
    """Example usage of the FinBERT sentiment pipeline."""
    # Initialize the pipeline
    pipeline = FinBERTSentimentPipeline()
    
    # Example texts
    example_texts = [
        "The company reported strong earnings growth in Q4.",
        "Shares plummeted after disappointing revenue results.",
        "The market remained stable throughout the trading session."
    ]
    
    print("FinBERT Sentiment Analysis Examples:\n")
    for text in example_texts:
        result = pipeline.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
        print(f"Scores: {result['scores']}\n")


if __name__ == "__main__":
    main()
