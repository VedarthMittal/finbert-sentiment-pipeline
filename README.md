# Overcoming the 512-Token FinBERT Limit in Earnings Call Analysis

**Applied AI Research Seminar (6414M0712Y)** 

---

## 1. Research Context: The Information Gap

FinBERT, the industry-standard model for financial sentiment, is constrained by a **hard 512-token limit**. However, corporate earnings call transcripts average **~7,349 tokens**, creating a significant barrier for automated analysis.



* **Truncation Bias**: Naive truncation (cutting at 512 tokens) discards approximately 93% of the document.
* **Missing Q&A**: Crucial spontaneous dialogue (Q&A) typically occurs late in the call and is systematically lost in standard processing.
* **The Goal**: This project utilizes **budget-aware extractive summarization** to compress transcripts by 94% while preserving categorical sentiment signals.

---

## 2. Methodology: Budget-Aware Summarization

### Algorithm Design


We implement a hybrid ranking and selection system:
* **Sentence Ranking**: A combination of **TF-IDF** (statistical importance) and **TextRank** (semantic centrality) with a 1.2x **Keyword Boost** for financial indicators (*e.g., guidance, outlook, margin, EBITDA*).
* **Segmented Budgeting**: 
    * **150 Tokens** allocated to Prepared Remarks (scripted content).
    * **300 Tokens** allocated to Q&A (prioritizing spontaneous management insights).
* **Total Budget**: 450 Tokens (ensuring a safety buffer for the 512-token FinBERT sub-word tokenizer).

---

## 3. Requirements

### Hardware Requirements
* **GPU**: NVIDIA GPU (8GB+ VRAM) recommended for Stage 4 (FinBERT Inference). Compatible with CUDA and Apple Silicon (MPS).
* **RAM**: 16GB minimum (32GB recommended for large-scale data processing).
* **Storage**: 5GB+ for model checkpoints and intermediate data artifacts.

### Software Prerequisites
* **Python**: 3.13
* **OpenAI API Key**: Required for Stage 5 (Ground Truth Generation).

### Python Dependencies
Install the core stack via the provided command in the Setup section:
* `transformers`, `torch`, `scikit-learn`, `networkx`, `nltk`, `openai`, `python-dotenv`, `pandas`, `numpy`, `matplotlib`, `seaborn`
---

## 4. Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/VedarthMittal/finbert-summarization-pipeline.git](https://github.com/VedarthMittal/finbert-summarization-pipeline.git)
cd finbert-summarization-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn transformers torch networkx tqdm matplotlib seaborn nltk python-dotenv openai

# Download NLTK resources
python -c "import nltk; nltk.download('punkt')"

# Configure Environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

---
## 5. Pipeline Execution

Execute the stages in sequential order to maintain data integrity:

| Stage | Script | Description |
| :--- | :--- | :--- |
| **1** | `eda_information_gap.py` | Quantifies the "Information Gap" and token loss statistics. |
| **2** | `preprocessing_stage2.py` | Performs NLTK sentence tokenization and regex cleaning. |
| **3** | `stage3_budget_aware_summarization.py` | Runs the TF-IDF/TextRank budget-aware summarizer. |
| **4** | `stage4_gpu_optimized.py` | Performs batch FinBERT inference (Summary vs. Baseline). |
| **5** | `stage5_ground_truth_llm.py` | Generates GPT-4o proxy-expert labels with rationales. |
| **6** | `stage6_evaluation_metrics.py` | Generates final F1-scores, Confusion Matrices, and charts. |

---

## 6. Evaluation Framework

We validate the pipeline using a **Triangulation Approach**:

1. **Baseline**: Full-text sentiment via a sliding-window average (7,000+ tokens).
2. **Experimental**: Sentiment on our condensed 450-token extractive summary.
3. **Ground Truth**: Expert-level annotations from GPT-4o on stratified model-disagreement cases.



---

## 7. Results Summary

* **Compression Ratio**: ~94.2% reduction in input size.
* **Compliance**: 100% compliance with FinBERT 512-token constraints.
* **Speedup**: ~25.6x reduction in inference time compared to sliding-window baselines.



---

## 8. Citation & License

This work is submitted as academic research for the Master's in Data Science and Business Amalytics at the **University of Amsterdam**.

```bibtex
@mastersthesis{finbert_summarization_2026,
  author = {[Vedarth Mittal]},
  title = {Overcoming the 512-Token FinBERT Limit in Earnings Call Analysis},
  school = {University of Amsterdam},
  year = {2026}
}
```

---
## 9. Contact

* **Research by**: Vedarth Mittal
* **Email**: mittalvedarth@student.uva.nl
* **GitHub**: VedarthMittal
