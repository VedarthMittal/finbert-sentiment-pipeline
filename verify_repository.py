"""
Stage 5: Ground Truth Generation (Robust Version)
Fixes the 'unhashable type' list error and matches your specific file names.
"""

import os
import json
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

load_dotenv()

# --- CONFIGURATION ---
INPUT_S3 = Path("stage3_output/extractive_summaries.pkl")
INPUT_S4 = Path("stage4_output/final_sentiment_results.pkl")
OUTPUT_DIR = Path("outputs")
API_KEY = os.getenv("OPENAI_API_KEY")

RANDOM_SEED = 42
SAMPLE_SIZE = 100
MAX_RETRIES = 3

if not API_KEY:
    print("❌ Error: OPENAI_API_KEY not found. Add it to your .env file.")
    sys.exit(1)

class GroundTruthEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.model = "gpt-4o"
        self.system_prompt = (
            "You are a Senior Equity Analyst. Analyze the following earnings call "
            "transcript summary and provide a sentiment label: POSITIVE, NEGATIVE, or NEUTRAL. "
            "Return JSON: {'label': 'STR', 'rationale': 'STR'}"
        )

    def load_and_align_data(self) -> pd.DataFrame:
        """Loads files and cleans 'unhashable' list columns for merging."""
        if not (INPUT_S3.exists() and INPUT_S4.exists()):
            raise FileNotFoundError(f"Missing: {INPUT_S3} or {INPUT_S4}")
            
        s3 = pd.read_pickle(INPUT_S3)
        s4 = pd.read_pickle(INPUT_S4)

        def sanitize_key(df, col):
            """Extracts first item if value is a list, then converts to string."""
            # This fixes the 'unhashable type: list' error
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
            df[col] = df[col].astype(str).str.strip()
            return df

        print("Sanitizing join keys (ticker/date)...")
        s3 = sanitize_key(s3, 'ticker')
        s3 = sanitize_key(s3, 'date')
        s4 = sanitize_key(s4, 'ticker')
        s4 = sanitize_key(s4, 'date')
        
        # Merge only necessary columns to prevent memory bloat
        return pd.merge(
            s4[['ticker', 'date', 'summary_label', 'baseline_label']], 
            s3[['ticker', 'date', 'extractive_summary']], 
            on=['ticker', 'date'], 
            how='inner'
        )

    def call_llm(self, text: str):
        """Robust API call with exponential backoff for rate limits."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": f"Analyze: {text[:6000]}"}],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                res = json.loads(response.choices[0].message.content)
                label = res.get('label', 'NEUTRAL').upper()
                if label not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    label = "NEUTRAL"
                return label, res.get('rationale', ''), True
            except RateLimitError:
                time.sleep(60 * (attempt + 1))
            except Exception as e:
                if attempt == MAX_RETRIES - 1: return "NEUTRAL", str(e), False
                time.sleep(5)
        return "NEUTRAL", "Timeout", False

    def run(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("Starting Stage 5...")
        
        df = self.load_and_align_data()
        
        # Stratified sampling (Agreement vs Disagreement)
        np.random.seed(RANDOM_SEED)
        dis = df[df["summary_label"] != df["baseline_label"]]
        agr = df[df["summary_label"] == df["baseline_label"]]
        n_dis = min(SAMPLE_SIZE // 2, len(dis))
        n_agr = SAMPLE_SIZE - n_dis
        
        sampled = pd.concat([
            dis.sample(n=n_dis, random_state=RANDOM_SEED),
            agr.sample(n=n_agr, random_state=RANDOM_SEED)
        ]).sample(frac=1, random_state=RANDOM_SEED)

        results = []
        for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="GPT-4o Evaluation"):
            label, rationale, success = self.call_llm(row['extractive_summary'])
            results.append({
                "ticker": row['ticker'],
                "date": row['date'],
                "summary_label": row['summary_label'],
                "baseline_label": row['baseline_label'],
                "llm_ground_truth": label,
                "llm_rationale": rationale,
                "success": success
            })

        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_DIR / "05_ground_truth_eval.csv", index=False)
        print(f"Success! Processed {len(res_df)} samples. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    GroundTruthEvaluator().run()