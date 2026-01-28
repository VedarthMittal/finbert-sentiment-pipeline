"""
Stage 2: Sentence Extraction

Converting raw transcripts into sentence lists for extractive summarization.
Need sentence-level granularity since TF-IDF and TextRank work on sentences, not words.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import nltk
from nltk.tokenize import sent_tokenize
import re
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("Stage 2: Sentence Extraction")
print("="*60)

# NLTK punkt tokenizer needed for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

DATA_PATH = Path("data/motley-fool-data.pkl")
OUTPUT_DIR = Path("preprocessing_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Input: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}\n")

# Load data
transcripts_df = pd.read_pickle(DATA_PATH)
print(f"Loaded {len(transcripts_df):,} transcripts")

# Check if already segmented from Stage 1
if 'remarks' not in transcripts_df.columns or 'q_and_a' not in transcripts_df.columns:
    print("Re-segmenting transcripts...")
    
    text_col = None
    for col in ['transcript', 'text', 'content', 'body']:
        if col in transcripts_df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Can't find transcript column")
    
    def segment_transcript(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "", ""
        
        qa_patterns = [
            r'(?:Questions?\s+(?:and|&)\s+Answers?)',
            r'(?:Q\s*&\s*A)',
            r'(?:^Questions?\s+Session)',
        ]
        
        qa_match = None
        for pattern in qa_patterns:
            qa_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if qa_match:
                break
        
        if qa_match:
            split_pos = qa_match.start()
            return text[:split_pos].strip(), text[split_pos:].strip()
        else:
            return text.strip(), ""
    
    transcripts_df[['remarks', 'q_and_a']] = transcripts_df[text_col].apply(
        lambda x: pd.Series(segment_transcript(x))
    )

# Clean sentences - remove operator noise but keep everything else
# NOT removing stop words because TF-IDF needs full context

def clean_sentence(text):
    if not isinstance(text, str):
        return ""
    
    # Remove [Operator Instructions] and similar
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove common operator phrases that don't add info
    text = re.sub(r'operator\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:good\s+)?morning\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'thank\s+you(?:\s+for\s+your\s+question)?', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Tokenize into sentences
def tokenize_sentences(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    
    try:
        raw_sents = sent_tokenize(text)
        cleaned = [clean_sentence(s) for s in raw_sents]
        # Filter out empty strings after cleaning
        return [s for s in cleaned if len(s) > 0]
    except Exception as e:
        print(f"Error tokenizing: {e}")
        return []

print("Tokenizing remarks...")
transcripts_df['remarks_sentences'] = transcripts_df['remarks'].apply(tokenize_sentences)

print("Tokenizing Q&A...")
transcripts_df['qa_sentences'] = transcripts_df['q_and_a'].apply(tokenize_sentences)

# Stats
transcripts_df['n_remarks_sentences'] = transcripts_df['remarks_sentences'].apply(len)
transcripts_df['n_qa_sentences'] = transcripts_df['qa_sentences'].apply(len)
transcripts_df['n_total_sentences'] = transcripts_df['n_remarks_sentences'] + transcripts_df['n_qa_sentences']

print(f"\nSentences per transcript:")
print(f"  Remarks - mean: {transcripts_df['n_remarks_sentences'].mean():.0f}, median: {transcripts_df['n_remarks_sentences'].median():.0f}")
print(f"  Q&A - mean: {transcripts_df['n_qa_sentences'].mean():.0f}, median: {transcripts_df['n_qa_sentences'].median():.0f}")
print(f"  Total corpus: {transcripts_df['n_total_sentences'].sum():,} sentences")

remarks_total = transcripts_df['n_remarks_sentences'].sum()
qa_total = transcripts_df['n_qa_sentences'].sum()
print(f"\nBreakdown: {remarks_total:,} remarks ({remarks_total/(remarks_total+qa_total)*100:.0f}%), {qa_total:,} Q&A ({qa_total/(remarks_total+qa_total)*100:.0f}%)")

# Quick quality check - inspect one transcript
qa_with_sents = transcripts_df[transcripts_df['n_qa_sentences'] > 0]

if len(qa_with_sents) > 0:
    sample = qa_with_sents.iloc[0]
    
    print(f"\nSample transcript:")
    print(f"Remarks: {sample['n_remarks_sentences']} sentences")
    print(f"Q&A: {sample['n_qa_sentences']} sentences")
    
    print(f"\nFirst 3 remarks sentences:")
    for i, s in enumerate(sample['remarks_sentences'][:3], 1):
        print(f"  {i}. {s}")
    
    print(f"\nFirst 3 Q&A sentences:")
    for i, s in enumerate(sample['qa_sentences'][:3], 1):
        print(f"  {i}. {s}")

# Export
output_full = OUTPUT_DIR / 'transcripts_with_sentences.pkl'
transcripts_df.to_pickle(output_full)
print(f"\nSaved: {output_full}")

# Lightweight version for Stage 3
df_lite = transcripts_df[['ticker', 'date', 'remarks_sentences', 'qa_sentences', 
                           'n_remarks_sentences', 'n_qa_sentences', 'n_total_sentences']].copy()
output_lite = OUTPUT_DIR / 'transcripts_sentences_only.pkl'
df_lite.to_pickle(output_lite)
print(f"Saved: {output_lite}")

# Stats CSV
df_stats = transcripts_df[['ticker', 'date', 'n_remarks_sentences', 'n_qa_sentences', 'n_total_sentences']].copy()
output_csv = OUTPUT_DIR / 'tokenization_stats.csv'
df_stats.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")

print(f"\nDone. Ready for Stage 3 (extractive summarization).\n")
