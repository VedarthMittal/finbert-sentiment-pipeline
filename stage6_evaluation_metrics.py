"""
Stage 6: Statistical Evaluation & Visualization
Calculates final performance metrics against LLM ground truth and generates 
publication-quality visualizations for methodological validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


INPUT_GT = Path("stage5_output/05_ground_truth_eval.pkl")
OUTPUT_DIR = Path("stage6_output")
PLOTS_DIR = OUTPUT_DIR / "plots"
LABELS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

class EvaluationPipeline:
    def __init__(self):
        # Set academic plotting style
        sns.set_theme(style="white", palette="muted")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'sans-serif'
        OUTPUT_DIR.mkdir(exist_ok=True)
        PLOTS_DIR.mkdir(exist_ok=True)

    def load_clean_data(self) -> pd.DataFrame:
        """Load ground truth and handle missing success columns gracefully."""
        # Try both the specific output folder and the local directory
        possible_paths = [INPUT_GT, Path("05_ground_truth_eval.pkl")]
        df = None
        
        for path in possible_paths:
            if path.exists():
                df = pd.read_pickle(path)
                break
        
        if df is None:
            raise FileNotFoundError("Ground truth file (05_ground_truth_eval.pkl) not found.")
        
        # FIX: Check if 'success' column exists before filtering to avoid KeyError
        if 'success' in df.columns:
            initial_len = len(df)
            df = df[df['success'] == True].copy()
            print(f"Filtered {len(df)} successful samples (Dropped {initial_len - len(df)} failures).")
        else:
            print(f"Proceeding with {len(df)} samples (No 'success' column found to filter).")
            
        return df

    def get_metrics(self, y_true, y_pred, name):
        """Standardize classification metrics calculation."""
        # Ensure all labels are uppercase strings for comparison
        y_true = y_true.astype(str).str.upper()
        y_pred = y_pred.astype(str).str.upper()
        
        f1_pc = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
        
        return {
            'model': name,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_positive': f1_pc[0],
            'f1_negative': f1_pc[1],
            'f1_neutral': f1_pc[2],
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def plot_results(self, df, summary_m, baseline_m):
        """Generate confusion matrices and F1 comparison charts."""
        y_true = df['llm_ground_truth'].astype(str).str.upper()
        
        # 1. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, (col, title) in enumerate([('summary_label', 'Extractive Summary'), 
                                         ('baseline_label', 'Sliding Window')]):
            y_pred = df[col].astype(str).str.upper()
            cm = confusion_matrix(y_true, y_pred, labels=LABELS)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i],
                        xticklabels=LABELS, yticklabels=LABELS, cbar=False)
            axes[i].set_title(title, fontweight='bold')
            axes[i].set_ylabel('Ground Truth')
            axes[i].set_xlabel('Predicted')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrices.png")

        # 2. Performance Comparison Bar Chart
        
        metrics_to_plot = ['accuracy', 'f1_macro', 'f1_neutral']
        labels_plot = ['Accuracy', 'F1-Macro', 'F1-Neutral']
        
        x = np.arange(len(labels_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [summary_m[m] for m in metrics_to_plot], width, label='Summary', color='#BC0031')
        ax.bar(x + width/2, [baseline_m[m] for m in metrics_to_plot], width, label='Baseline', color='#707070')
        
        ax.set_ylabel('Score')
        ax.set_title('Methodological Comparison: Summary vs. Sliding Window')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot)
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        plt.savefig(PLOTS_DIR / "performance_comparison.png")

    def run(self):
        print("--- Stage 6: Statistical Evaluation ---")
        try:
            df = self.load_clean_data()
            y_gt = df['llm_ground_truth'].astype(str).str.upper()

            s_metrics = self.get_metrics(y_gt, df['summary_label'], 'Extractive Summary')
            b_metrics = self.get_metrics(y_gt, df['baseline_label'], 'Sliding Window')

            self.plot_results(df, s_metrics, b_metrics)

            # Export Results
            metrics_df = pd.DataFrame([s_metrics, b_metrics])
            metrics_df.to_csv(OUTPUT_DIR / "final_metrics_comparison.csv", index=False)
            
            print("\nFinal Performance Results:")
            print(metrics_df[['model', 'accuracy', 'f1_macro']].to_string(index=False))
            print(f"\nEvaluation complete. Files saved to {OUTPUT_DIR.absolute()}")
        
        except Exception as e:
            print(f"❌ Error in Stage 6: {e}")

if __name__ == "__main__":
    EvaluationPipeline().run()
