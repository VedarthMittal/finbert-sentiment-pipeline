"""
Stage 4: Sentiment Inference & Methodological Validation
Performs batch sentiment analysis on extractive summaries and validates 
them against a sliding-window baseline for information preservation.

"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_SUMMARIES = Path("stage3_output/extractive_summaries.pkl")
INPUT_RAW_DATA = Path("data/motley-fool-data.pkl")
OUTPUT_DIR = Path("stage4_output")
MODEL_NAME = "yiyanghkust/finbert-tone"
TEXT_COLUMN = 'transcript'

# Inference Settings
BATCH_SIZE = 32
MAX_LENGTH = 512
BASELINE_SAMPLE_SIZE = 1000
SLIDING_WINDOW_SIZE = 512
SLIDING_WINDOW_OVERLAP = 50

# HuggingFace Cache Management
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_compute_environment():
    """Report hardware capabilities for reproducibility."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Hardware: GPU - {name} ({mem:.1f} GB Available)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Hardware: Apple Silicon MPS")
        return torch.device("mps")
    print("Hardware: CPU")
    return torch.device("cpu")

def get_sentiment_batch(texts, tokenizer, model, device, desc="Inference", disable_tqdm=False):
    """Batch sentiment inference with confidence scores and probability distributions."""
    results = []
    class_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with torch.no_grad():
        # disable_tqdm=True prevents nested bars during sliding window analysis
        for i in tqdm(range(0, len(texts), BATCH_SIZE), total=num_batches, desc=desc, disable=disable_tqdm):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, max_length=MAX_LENGTH, truncation=True, 
                               padding=True, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            scores, preds = torch.max(probs, dim=1)
            
            probs_np = probs.cpu().numpy()
            scores_np = scores.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            for j in range(len(batch)):
                results.append({
                    'label': class_names[preds_np[j]],
                    'score': float(scores_np[j]),
                    'probs': {class_names[k]: float(probs_np[j, k]) for k in range(3)}
                })
    return results

def sliding_window_inference(text, tokenizer, model, device):
    """Aggregates sentiment across overlapping windows for long-context validation."""
    if not isinstance(text, str) or not text.strip():
        return {'label': 'NEUTRAL', 'score': 0.0, 'n_chunks': 0, 
                'probs': {'NEGATIVE': 0.0, 'NEUTRAL': 1.0, 'POSITIVE': 0.0}}
    
    tokens = tokenizer.tokenize(text)
    stride = SLIDING_WINDOW_SIZE - SLIDING_WINDOW_OVERLAP
    windows = [tokenizer.convert_tokens_to_string(tokens[s:s+SLIDING_WINDOW_SIZE]) 
               for s in range(0, len(tokens), stride)]
    
    if not windows:
        return {'label': 'NEUTRAL', 'score': 0.0, 'n_chunks': 0, 
                'probs': {'NEGATIVE': 0.0, 'NEUTRAL': 1.0, 'POSITIVE': 0.0}}
    
    # disable_tqdm=True is critical here to avoid terminal flooding
    results = get_sentiment_batch(windows, tokenizer, model, device, disable_tqdm=True)
    
    avg_probs = {c: np.mean([r['probs'][c] for r in results]) for c in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']}
    label = max(avg_probs, key=avg_probs.get)
    return {'label': label, 'score': float(avg_probs[label]), 'n_chunks': len(windows), 'probs': avg_probs}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_compute_environment()

    # Model Initialization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR)).to(device)
    model.eval()

    # Data Loading
    summ_df = pd.read_pickle(INPUT_SUMMARIES)
    raw_df = pd.read_pickle(INPUT_RAW_DATA)
    summ_df['raw_text'] = raw_df[TEXT_COLUMN].values[:len(summ_df)]

    # 1. Experimental Pipeline: Summaries
    print(f"\nPhase 1: Analyzing {len(summ_df):,} extractive summaries...")
    start_summ = time.time()
    results_summ = get_sentiment_batch(summ_df['extractive_summary'].tolist(), tokenizer, model, device, desc="Summary Inference")
    time_summ = time.time() - start_summ
    
    summ_df['summary_label'] = [r['label'] for r in results_summ]
    summ_df['summary_score'] = [r['score'] for r in results_summ]
    summ_df['summary_probs'] = [r['probs'] for r in results_summ]

    # 2. Baseline Pipeline: Sliding Window
    base_df = summ_df.iloc[:BASELINE_SAMPLE_SIZE].copy()
    print(f"\nPhase 2: Analyzing baseline (n={BASELINE_SAMPLE_SIZE}) with sliding windows...")
    start_base = time.time()
    
    results_base = []
    for txt in tqdm(base_df['raw_text'], desc="Sliding Window Baseline"):
        results_base.append(sliding_window_inference(txt, tokenizer, model, device))
    
    time_base = time.time() - start_base
    base_df['baseline_label'] = [r['label'] for r in results_base]
    base_df['baseline_score'] = [r['score'] for r in results_base]
    base_df['baseline_probs'] = [r['probs'] for r in results_base]
    base_df['baseline_n_chunks'] = [r['n_chunks'] for r in results_base]

    # 3. Methodological Validation Metrics
    base_df['labels_agree'] = base_df['summary_label'] == base_df['baseline_label']
    base_df['score_diff'] = abs(base_df['summary_score'] - base_df['baseline_score'])
    
    r, p_val = pearsonr(base_df['summary_score'], base_df['baseline_score'])
    acc = accuracy_score(base_df['baseline_label'], base_df['summary_label'])
    
    speed_summ = time_summ / len(summ_df)
    speed_base = time_base / len(base_df)
    speedup = speed_base / speed_summ

    # 4. Performance Summary
    results_table = pd.DataFrame({
        'Metric': ['Pearson r Correlation', 'Categorical Accuracy', 'Inference Speedup', 'Estimated Time Saved'],
        'Value': [f"{r:.4f}", f"{acc*100:.2f}%", f"{speedup:.1f}x", f"{(speed_base * len(summ_df) - time_summ)/60:.1f} minutes"]
    })
    print("\n" + "="*40 + "\nVALIDATION SUMMARY\n" + "="*40)
    print(results_table.to_string(index=False))

    # 5. Export
    export_cols = ['ticker', 'date', 'summary_label', 'summary_score', 'summary_probs',
                   'baseline_label', 'baseline_score', 'baseline_probs', 'labels_agree', 
                   'score_diff', 'baseline_n_chunks', 'extractive_summary']
    
    base_df[export_cols].to_pickle(OUTPUT_DIR / 'final_sentiment_results.pkl')
    base_df[export_cols].to_csv(OUTPUT_DIR / 'final_sentiment_results.csv', index=False)
    print(f"\nInference complete. High-fidelity results exported to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
