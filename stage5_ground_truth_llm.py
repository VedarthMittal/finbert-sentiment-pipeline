# GPT-4o ground truth generation for sentiment validation

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI SDK
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# Configure paths
WORKSPACE_ROOT = Path.cwd()
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("❌ Error: OPENAI_API_KEY not found!")
    print("\nPlease set your OpenAI API key using one of these methods:")
    print("  1. Create a .env file with: OPENAI_API_KEY=your-key-here")
    print("  2. Set environment variable: export OPENAI_API_KEY=your-key-here")
    print("  3. Windows PowerShell: $env:OPENAI_API_KEY='your-key-here'")
    sys.exit(1)


@dataclass
class GroundTruthResult:
    ticker: str
    date: str
    summary_label: str
    baseline_label: str
    llm_ground_truth: str
    llm_rationale: str
    api_call_time: float
    success: bool
    error_message: Optional[str] = None


class GroundTruthEvaluator:
    # GPT-4o ground truth label generator
    
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.results: List[GroundTruthResult] = []
        self.model = "gpt-4o"
        
        self.system_prompt = (
            "You are a Senior Equity Analyst. Analyze the following earnings call "
            "transcript summary and provide a sentiment label: POSITIVE, NEGATIVE, or NEUTRAL. "
            "A POSITIVE label means the company exceeds expectations or provides strong guidance. "
            "NEGATIVE means headwinds, risks, or missed targets. NEUTRAL is for standard "
            "procedural updates with no clear direction. "
            "Return your response in JSON format with two fields: 'label' (POSITIVE/NEGATIVE/NEUTRAL) "
            "and 'rationale' (your reasoning)."
        )
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load Stage 4 and Stage 3 outputs
        print("\n Loading Stage 4 and Stage 3 data...")
        
        # Load Stage 4 final inference results
        stage4_file = OUTPUTS_DIR / "04_final_inference.pkl"
        if not stage4_file.exists():
            raise FileNotFoundError(f"Missing: {stage4_file}")
        
        stage4_data = pd.read_pickle(stage4_file)
        print(f"  Stage 4 columns: {stage4_data.columns.tolist()}")
        print(f"  Stage 4 shape: {stage4_data.shape}")
        
        # Load Stage 3 summaries
        stage3_file = OUTPUTS_DIR / "03_summaries.pkl"
        if not stage3_file.exists():
            raise FileNotFoundError(f"Missing: {stage3_file}")
        
        stage3_data = pd.read_pickle(stage3_file)
        print(f"  Stage 3 columns: {stage3_data.columns.tolist()}")
        print(f"  Stage 3 shape: {stage3_data.shape}")
        
        print(f"  Loaded {len(stage4_data)} transcripts from Stage 4")
        print(f"  Loaded {len(stage3_data)} summaries from Stage 3")
        
        return stage4_data, stage3_data
    
    def stratified_sample(self, stage4_df: pd.DataFrame, sample_size: int = 100) -> List[int]:
        # 50/50 split between agreement and disagreement cases
        print(f"\n Performing stratified sampling ({sample_size} transcripts)...")
        
        # Identify disagreement cases
        disagreement_mask = stage4_df["summary_label"] != stage4_df["baseline_label"]
        disagreement_indices = stage4_df[disagreement_mask].index.tolist()
        agreement_indices = stage4_df[~disagreement_mask].index.tolist()
        
        print(f"  Disagreement cases: {len(disagreement_indices)}")
        print(f"  Agreement cases: {len(agreement_indices)}")
        
        # Sample with 50/50 split
        n_disagreement = min(sample_size // 2, len(disagreement_indices))
        n_agreement = min(sample_size - n_disagreement, len(agreement_indices))
        
        sampled_disagreement = np.random.choice(
            disagreement_indices, 
            size=n_disagreement,
            replace=False
        )
        sampled_agreement = np.random.choice(
            agreement_indices,
            size=n_agreement,
            replace=False
        )
        
        sampled_indices = list(sampled_disagreement) + list(sampled_agreement)
        np.random.shuffle(sampled_indices)
        
        print(f"  Sampled {len(sampled_indices)} transcripts")
        print(f"    - Disagreement: {len(sampled_disagreement)}")
        print(f"    - Agreement: {len(sampled_agreement)}")
        
        return sampled_indices
    
    def call_llm(self, summary_text: str, max_retries: int = 3) -> Tuple[str, str, float, bool]:
        # GPT-4o API call with retry logic
        start_time = time.time()
        
        user_prompt = f"Analyze this earnings call summary:\n\n{summary_text}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=200,
                    timeout=30.0
                )
                
                api_time = time.time() - start_time
                
                # Parse JSON response
                try:
                    result = json.loads(response.choices[0].message.content)
                    label = result.get("label", "NEUTRAL").upper()
                    rationale = result.get("rationale", "No rationale provided")
                    
                    # Validate label
                    if label not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                        label = "NEUTRAL"
                    
                    return label, rationale, api_time, True
                
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return "NEUTRAL", "JSON parse error", api_time, False
            
            except RateLimitError as e:
                wait_time = 60 * (attempt + 1)
                print(f"  Warning: Rate limit (attempt {attempt+1}), waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return "NEUTRAL", f"Rate limit exceeded after {max_retries} retries", time.time() - start_time, False
            
            except APIConnectionError as e:
                print(f"  Warning: Connection error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return "NEUTRAL", f"API connection failed: {str(e)[:100]}", time.time() - start_time, False
            
            except APIError as e:
                print(f"  Warning: API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return "NEUTRAL", f"API error: {str(e)[:100]}", time.time() - start_time, False
        
        api_time = time.time() - start_time
        return "NEUTRAL", "Max retries exceeded", api_time, False
    
    def process_samples(
        self, 
        stage4_df: pd.DataFrame, 
        stage3_df: pd.DataFrame,
        sampled_indices: List[int]
    ) -> pd.DataFrame:
        # Process samples through GPT-4o
        print(f"\nProcessing {len(sampled_indices)} samples through GPT-4o...")
        print(f"(Estimating ~{len(sampled_indices) * 3} seconds with rate limits)\n")
        
        results = []
        total_api_time = 0
        failed_count = 0
        
        for i, idx in enumerate(sampled_indices, 1):
            # Get metadata from Stage 4
            row_stage4 = stage4_df.iloc[idx]
            ticker = row_stage4.get("ticker", f"TICKER_{idx}")
            date = row_stage4.get("date", f"DATE_{idx}")
            summary_label = row_stage4.get("summary_label", "UNKNOWN")
            baseline_label = row_stage4.get("baseline_label", "UNKNOWN")
            
            # Get summary text from Stage 3
            row_stage3 = stage3_df.iloc[idx]
            summary_text = row_stage3.get("extractive_summary", "")
            
            if not summary_text or not isinstance(summary_text, str):
                print(f"  [{i:3d}/{len(sampled_indices)}] {ticker} - Warning: SKIP (no summary)")
                failed_count += 1
                continue
            
            # Truncate if too long (to avoid token limits)
            if len(summary_text) > 1500:
                summary_text = summary_text[:1500] + "..."
            
            # Call LLM
            label, rationale, api_time, success = self.call_llm(summary_text)
            total_api_time += api_time
            
            # Status indicator
            status = "OK" if success else "Failed"
            
            print(f"  [{i:3d}/{len(sampled_indices)}] {ticker} ({str(date)[:10]}) - {status} {label} ({api_time:.1f}s)")
            
            if not success:
                failed_count += 1
            
            # Store result
            result = GroundTruthResult(
                ticker=ticker,
                date=str(date),
                summary_label=summary_label,
                baseline_label=baseline_label,
                llm_ground_truth=label,
                llm_rationale=rationale,
                api_call_time=api_time,
                success=success,
                error_message=None if success else rationale
            )
            results.append(result)
        
        print(f"\n  Processed {len(results)} samples")
        print(f"  Successful: {len(results) - failed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total API time: {total_api_time:.1f}s ({total_api_time/60:.1f}m)")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                "ticker": r.ticker,
                "date": r.date,
                "summary_label": r.summary_label,
                "baseline_label": r.baseline_label,
                "llm_ground_truth": r.llm_ground_truth,
                "llm_rationale": r.llm_rationale,
            }
            for r in results
        ])
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        # Save results to pickle and CSV
        print("\n Saving ground truth results...")
        
        # Pickle
        pkl_file = OUTPUTS_DIR / "05_ground_truth_eval.pkl"
        results_df.to_pickle(pkl_file)
        print(f"  Saved: {pkl_file.name}")
        
        # CSV for readability
        csv_file = OUTPUTS_DIR / "05_ground_truth_eval.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"  Saved: {csv_file.name}")
    
    def generate_report(self, results_df: pd.DataFrame) -> None:
        # Summary report
        print("\n" + "="*70)
        print("")
        print("="*70)
        
        # Label distribution
        print("\n LLM Ground Truth Labels:")
        label_dist = results_df["llm_ground_truth"].value_counts()
        for label, count in label_dist.items():
            pct = 100 * count / len(results_df)
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Agreement metrics
        print("\n Comparison with Stage 4 Models:")
        
        summary_agree = (results_df["summary_label"] == results_df["llm_ground_truth"]).sum()
        baseline_agree = (results_df["baseline_label"] == results_df["llm_ground_truth"]).sum()
        both_agree = (
            (results_df["summary_label"] == results_df["llm_ground_truth"]) &
            (results_df["baseline_label"] == results_df["llm_ground_truth"])
        ).sum()
        
        print(f"  Summary agrees with LLM: {summary_agree}/{len(results_df)} ({100*summary_agree/len(results_df):.1f}%)")
        print(f"  Baseline agrees with LLM: {baseline_agree}/{len(results_df)} ({100*baseline_agree/len(results_df):.1f}%)")
        print(f"  Both agree with LLM: {both_agree}/{len(results_df)} ({100*both_agree/len(results_df):.1f}%)")
        
        # Preview first 3
        print("\n First 3 Samples with Rationales:")
        
        for i, (idx, row) in enumerate(results_df.head(3).iterrows(), 1):
            print(f"\n{i}. {row['ticker']} ({row['date'][:10]})")
            print(f"   Summary Label:    {row['summary_label']}")
            print(f"   Baseline Label:   {row['baseline_label']}")
            print(f"   LLM Ground Truth: {row['llm_ground_truth']}")
            print(f"   Rationale: {row['llm_rationale'][:200]}...")
        
        print("\n" + "="*70)
        print("")
        print("="*70)
        print("""
This ground truth is generated by GPT-4o and serves as a PROXY for human expert 
annotation. It is NOT infallible and may contain hallucinations, especially in:

1. Ambiguous earnings calls with mixed signals
2. Technical financial jargon beyond training data
3. Future guidance predictions (inherently uncertain)

The stored rationales allow auditing for obvious hallucinations. For production
use, human expert review is still recommended, especially for high-stakes decisions.

However, as a cost-effective benchmark, this approach provides:
- Consistency (same system prompt for all samples)
- Reproducibility (documented prompts and model)
- Quantitative evaluation metrics for thesis validation
""")
        print("="*70 + "\n")
    
    def run(self) -> bool:
        # Full pipeline execution
        try:
            print("\n" + "="*70)
            print("Ground truth generation")
            print("="*70)
            print(f"Workspace: {WORKSPACE_ROOT}")
            print(f"Model: {self.model}")
            
            # Load data
            stage4_df, stage3_df = self.load_data()
            
            # Stratified sampling
            sampled_indices = self.stratified_sample(stage4_df, sample_size=100)
            
            # Process through LLM
            results_df = self.process_samples(stage4_df, stage3_df, sampled_indices)
            
            # Save results
            self.save_results(results_df)
            
            # Generate report
            self.generate_report(results_df)
            
            return True
        
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            return False
        except Exception as e:
            print(f"\nError: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    evaluator = GroundTruthEvaluator()
    success = evaluator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()