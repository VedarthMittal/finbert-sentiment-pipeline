"""
Stage 5: Ground Truth Generation via LLM
Generates a deterministic stratified sample for expert-proxy validation.
Prioritizes model disagreement cases (50/50 split) to maximize 
methodological divergence testing.
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

# Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
INPUT_S3 = Path("stage3_output/extractive_summaries.pkl")
INPUT_S4 = Path("stage4_output/final_sentiment_results.pkl")
OUTPUT_DIR = Path("stage5_output")

# Use environment variable or fallback to hardcoded 
API_KEY = os.getenv("OPENAI_API_KEY") or 'your-key-here'

RANDOM_SEED = 42
SAMPLE_SIZE = 100
MAX_RETRIES = 3

if not API_KEY or API_KEY == 'your-key-here':
    print("❌ Error: OPENAI_API_KEY not found. Please set it in your .env file.")
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
        """Merge pipeline inputs and fix unhashable list issues."""
        if not (INPUT_S3.exists() and INPUT_S4.exists()):
            # Detailed error message to help you debug paths
            raise FileNotFoundError(
                f"Required files not found!\n"
                f"Expected Stage 3: {INPUT_S3.absolute()}\n"
                f"Expected Stage 4: {INPUT_S4.absolute()}"
            )
            
        s3 = pd.read_pickle(INPUT_S3)
        s4 = pd.read_pickle(INPUT_S4)
        
        # FIX: Ensure date is a string to prevent matching errors
        s3['date'] = s3['date'].astype(str)
        s4['date'] = s4['date'].astype(str)
        
        # FIX: Only select 'ticker' and 'date' to merge on. 
        # This avoids trying to merge on columns containing lists (which are unhashable).
        return pd.merge(
            s4[['ticker', 'date', 'summary_label', 'baseline_label']], 
            s3[['ticker', 'date', 'extractive_summary']], 
            on=['ticker', 'date'],
            how='inner'
        )

    def get_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic sampling: 50% agreement cases, 50% disagreement cases."""
        np.random.seed(RANDOM_SEED)
        
        # Identify Disagreement Cases (Where models diverge)
        disagree = df[df["summary_label"] != df["baseline_label"]]
        agree = df[df["summary_label"] == df["baseline_label"]]
        
        n_dis = min(SAMPLE_SIZE // 2, len(disagree))
        n_agr = SAMPLE_SIZE - n_dis
        
        # Perform stratified selection
        idx_dis = np.random.choice(disagree.index, size=n_dis, replace=False)
        idx_agr = np.random.choice(agree.index, size=n_agr, replace=False)
        
        sampled = df.loc[list(idx_dis) + list(idx_agr)].copy()
        print(f"Sampling Strategy: {n_dis} disagreements, {n_agr} agreements.")
        
        return sampled.sample(frac=1, random_state=RANDOM_SEED)

    def call_llm_with_retry(self, text: str):
        """Call GPT-4o with differentiated backoff and strict label validation."""
        clean_text = text[:6000] # Safe context window truncation
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": f"Analyze: {clean_text}"}],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                res = json.loads(response.choices[0].message.content)
                
                label = res.get('label', 'NEUTRAL').upper()
                if label not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    label = "NEUTRAL"
                
                return label, res.get('rationale', ''), True, None
            
            except RateLimitError:
                wait = 60 * (attempt + 1)
                time.sleep(wait)
            except (APIConnectionError, APIError) as e:
                time.sleep(5 * (attempt + 1))
                if attempt == MAX_RETRIES - 1: return "NEUTRAL", str(e), False, str(e)
            except Exception as e:
                return "NEUTRAL", str(e), False, str(e)
        
        return "NEUTRAL", "Max retries exceeded", False, "Timeout"

    def run(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("--- Stage 5: Ground Truth Generation ---")
        
        try:
            df = self.load_and_align_data()
            sampled_df = self.get_stratified_sample(df)
            
            results, success_count = [], 0
            print(f"Generating Ground Truth for {len(sampled_df)} samples...")

            for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="GPT-4o Evaluation"):
                label, rationale, success, error_msg = self.call_llm_with_retry(row['extractive_summary'])
                
                if success: success_count += 1
                
                results.append({
                    "ticker": row['ticker'],
                    "date": row['date'],
                    "summary_label": row['summary_label'],
                    "baseline_label": row['baseline_label'],
                    "llm_ground_truth": label,
                    "llm_rationale": rationale,
                    "success": success,
                    "error_message": error_msg
                })

            res_df = pd.DataFrame(results)
            res_df.to_csv(OUTPUT_DIR / "05_ground_truth_eval.csv", index=False)
            res_df.to_pickle(OUTPUT_DIR / "05_ground_truth_eval.pkl")

            print(f"\nPhase 5 Complete:")
            print(f"  Processed: {len(res_df)} | Successful: {success_count} | Failed: {len(res_df)-success_count}")
            print(f"  Outputs saved to {OUTPUT_DIR.absolute()}")
            
        except Exception as e:
            print(f"❌ Critical Error in Stage 5: {e}")

if __name__ == "__main__":
    GroundTruthEvaluator().run()
