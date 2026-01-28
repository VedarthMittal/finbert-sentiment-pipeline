"""
FinBERT Sentiment Analysis Pipeline

This module provides a sentiment analysis pipeline using FinBERT,
a pre-trained NLP model for financial sentiment analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union


class FinBERTSentimentPipeline:
    """
    A pipeline for analyzing sentiment in financial text using FinBERT.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the FinBERT sentiment analysis pipeline.
        
        Args:
            model_name (str): The name of the FinBERT model to use.
                            Default is "ProsusAI/finbert"
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["positive", "negative", "neutral"]
        
        # Set model to evaluation mode
        self.model.eval()
        
    def analyze_sentiment(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Analyze sentiment of the given text(s).
        
        Args:
            text (str or List[str]): Single text or list of texts to analyze
            
        Returns:
            Dict or List[Dict]: Sentiment analysis results with label and scores
        """
        # Handle single text input
        if isinstance(text, str):
            return self._analyze_single(text)
        
        # Handle list of texts
        results = []
        for t in text:
            results.append(self._analyze_single(t))
        return results
    
    def _analyze_single(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Dictionary containing sentiment label and scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        probabilities = predictions.cpu().numpy()[0]
        
        # Get the predicted label
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.labels[predicted_idx]
        
        # Create result dictionary
        result = {
            "text": text,
            "sentiment": predicted_label,
            "confidence": float(probabilities[predicted_idx]),
            "scores": {
                label: float(prob)
                for label, prob in zip(self.labels, probabilities)
            }
        }
        
        return result
    
    def batch_analyze(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Analyze sentiment of multiple texts in batches for better performance.
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process each result in the batch
            probabilities = predictions.cpu().numpy()
            
            for j, text in enumerate(batch):
                probs = probabilities[j]
                predicted_idx = np.argmax(probs)
                predicted_label = self.labels[predicted_idx]
                
                result = {
                    "text": text,
                    "sentiment": predicted_label,
                    "confidence": float(probs[predicted_idx]),
                    "scores": {
                        label: float(prob)
                        for label, prob in zip(self.labels, probs)
                    }
                }
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    pipeline = FinBERTSentimentPipeline()
    
    # Single text analysis
    text = "The company's revenue increased by 20% this quarter."
    result = pipeline.analyze_sentiment(text)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    print(f"Scores: {result['scores']}")
    
    print("\n" + "="*80 + "\n")
    
    # Multiple texts analysis
    texts = [
        "The stock price fell sharply after the earnings report.",
        "Investors are optimistic about the company's future prospects.",
        "The market remained stable with no significant changes."
    ]
    
    results = pipeline.analyze_sentiment(texts)
    for r in results:
        print(f"Text: {r['text']}")
        print(f"Sentiment: {r['sentiment']} (confidence: {r['confidence']:.2%})")
        print()
