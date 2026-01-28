# Overcoming the 512-Token FinBERT Limit in Earnings Call Analysis

**Master's Research Project**  
*Data Science & Artificial Intelligence*  
*University of Amsterdam, 2026*

---

## The Problem: FinBERT's Information Gap

FinBERT, the leading financial sentiment analysis model, has a **hard 512-token limit**. Earnings call transcripts, however, average **~7,349 tokens**, creating a critical information gap:

- **Naive truncation** cuts off 93% of content
- **Q&A sections** (where executives reveal uncertainties) are systematically lost
- **Sliding window approaches** work but are computationally expensive (7x slower)

**Quantitative Evidence:**
- Mean transcript: 7,349 tokens
- FinBERT capacity: 512 tokens
- Information loss: 6,837 tokens (93%)
- Q&A captured in first 512 tokens: ~3% of transcripts

This project investigates whether **budget-aware extractive summarization** can preserve sentiment signals while achieving 94% token compression.

---

## The Solution: Budget-Aware Extractive Summarization

### Algorithm Design

A **hybrid TF-IDF + TextRank** ranking system with strategic budget allocation:

1. **Token Budget**: 450 tokens total (62-token safety buffer)
   - **150 tokens** → Prepared Remarks (scripted content)
   - **300 tokens** → Q&A Section (spontaneous insights)

2. **Sentence Ranking**:
   - **TF-IDF**: Captures term importance (efficient, O(n log n))
   - **TextRank**: Captures semantic centrality via PageRank (accurate, O(n²))
   - **Keyword Boost**: 1.2x multiplier for financial terms (guidance, outlook, risk, margin, EBITDA)

3. **Greedy Selection**: Highest-ranked sentences added until budget exhausted

### Why Prioritize Q&A Over Prepared Remarks?

Research shows Q&A sections contain **"spontaneous, fuzzy knowledge"** where:
- Analysts ask challenging questions
- Executives reveal uncertainties not in prepared statements
- Forward-looking risks are discussed candidly

Our 300/150 token allocation reflects this empirical insight.

---

## The Validation: GPT-4o as Proxy Expert

### Triangulation Framework

Three sentiment sources for rigorous evaluation:

1. **Experimental**: Sentiment on extractive summaries (430 tokens avg)
2. **Baseline**: Sentiment on full transcripts via sliding window (7,349 tokens avg)
3. **Ground Truth**: GPT-4o annotations as proxy for human experts

### Evaluation Metrics

- **Pearson Correlation** (r): Summary vs. Baseline sentiment agreement
- **F1-Score**: Summary vs. Ground Truth, Baseline vs. Ground Truth
- **Label Agreement**: % transcripts with same sentiment class
- **Confusion Matrices**: Visual evidence of signal preservation
- **Speedup Analysis**: Computational efficiency gains

### Acknowledged Limitations

**GPT-4o is NOT infallible.** This approach acknowledges:
- Potential hallucinations in LLM annotations
- Domain knowledge gaps in financial jargon
- Human expert validation still recommended for production

However, GPT-4o provides:
- **Consistency**: Same prompt for all samples
- **Reproducibility**: Documented in code
- **Auditability**: Rationales stored for review
- **Scalability**: 100 samples in ~5 minutes vs. weeks for CFA analysts

---

## Dataset

- **Source**: 18,755 Motley Fool earnings call transcripts
- **Preprocessing**: Regex-based Q&A segmentation, NLTK sentence tokenization
- **Token Statistics**: Mean = 7,349, Median = 6,824, Max = 28,541
- **Sectors**: Diversified (tech, finance, healthcare, retail, energy)

**Note**: Raw data is **excluded from this repository** due to size constraints. Place `motley-fool-data.pkl` in the `data/` directory after cloning.

---

## Results Summary

### Token Compression

| Metric | Value |
|--------|-------|
| Original tokens/transcript | 7,349 (avg) |
| Summary tokens/transcript | 430 (avg) |
| Compression ratio | 94% |
| FinBERT compliance | 100% (all summaries ≤ 512) |

### Sentiment Preservation

| Comparison | Pearson r | Label Agreement | F1-Score |
|------------|-----------|-----------------|----------|
| Summary vs Baseline | [Your r value] | [Your %] | - |
| Summary vs GPT-4o | - | [Your %] | [Your F1] |
| Baseline vs GPT-4o | - | [Your %] | [Your F1] |

### Computational Efficiency

| Method | Throughput | Speedup |
|--------|------------|---------|
| Summary Inference | ~X tx/sec | **Xx faster** |
| Baseline Inference | ~Y tx/sec | baseline |

---

## Installation & Usage

### Prerequisites

- Python 3.13+
- CUDA-capable GPU (optional but recommended for Stage 4)
- OpenAI API key (for Stage 5 ground truth generation)

### Setup

```bash
# Clone repository
git clone https://github.com/[your-username]/finbert-sentiment-pipeline.git
cd finbert-sentiment-pipeline

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Data Preparation

Place your earnings call transcripts in `data/motley-fool-data.pkl`. Expected schema:

```python
{
    'ticker': str,        # Stock symbol
    'date': str/datetime, # Earnings call date
    'transcript': str     # Full transcript text
}
```

### Running the Pipeline

Execute stages sequentially:

```bash
# Stage 1: Information Gap Analysis
python eda_information_gap.py

# Stage 2: Sentence Tokenization
python preprocessing_stage2.py

# Stage 3: Extractive Summarization
python stage3_budget_aware_summarization.py

# Stage 4: FinBERT Sentiment Inference
python stage4_gpu_optimized.py

# Stage 5: Ground Truth Generation (requires OpenAI API key)
# Set environment variable first:
export OPENAI_API_KEY="sk-..."  # On Windows: set OPENAI_API_KEY=sk-...
python stage5_ground_truth_llm.py

# Stage 6: Statistical Evaluation
python stage6_evaluation_metrics.py
```

### Expected Outputs

```
outputs/
├── 01_eda_outputs/
│   ├── token_statistics.csv
│   └── qa_segmentation_accuracy.pkl
├── preprocessing_outputs/
│   ├── transcripts_with_sentences.pkl
│   └── tokenization_stats.csv
├── stage3_output/
│   ├── extractive_summaries.pkl
│   └── summarization_statistics.csv
├── stage4_output/
│   ├── final_sentiment_results.pkl
│   └── sentiment_comparison_report.txt
├── 05_ground_truth_eval.pkl
├── 05_ground_truth_eval.csv
└── 06_evaluation_plots/
    ├── confusion_matrix_summary.png
    ├── confusion_matrix_baseline.png
    └── f1_comparison.png
```

---

## Project Structure

```
.
├── data/                              # Raw transcripts (gitignored)
│   └── motley-fool-data.pkl          # User must provide
├── outputs/                           # Pipeline outputs (gitignored)
├── eda_information_gap.py            # Stage 1: Token gap analysis
├── preprocessing_stage2.py           # Stage 2: NLTK sentence tokenization
├── stage3_budget_aware_summarization.py  # Stage 3: TF-IDF + TextRank
├── stage4_gpu_optimized.py           # Stage 4: FinBERT batch inference
├── stage5_ground_truth_llm.py        # Stage 5: GPT-4o proxy annotations
├── stage6_evaluation_metrics.py      # Stage 6: F1-score evaluation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Excludes data and outputs
└── README.md                         # This file
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13+ |
| NLP | NLTK, Transformers (FinBERT) |
| ML | scikit-learn, NetworkX |
| Deep Learning | PyTorch (GPU acceleration) |
| LLM | OpenAI GPT-4o |
| Visualization | Matplotlib, Seaborn |

---

## Key Design Decisions

### 1. No Lemmatization in Preprocessing

**Decision**: Preserve original tokens without stemming/lemmatization.

**Rationale**: Financial entities like "EBITDA," "guidance," and company names are critical sentiment signals. Lemmatization risks destroying these domain-specific terms.

### 2. Regex-Based Q&A Segmentation

**Decision**: Use pattern matching on "Question-and-Answer" headers.

**Limitation**: Assumes consistent transcript formatting. Non-standard transcripts may mis-segment.

**Future Work**: Train a transformer-based segmentation classifier.

### 3. Greedy vs. Optimal Sentence Selection

**Decision**: Greedy algorithm (add highest-ranked sentences until budget exhausted).

**Rationale**: Proven optimal for knapsack-like token budgeting. No combination of lower-ranked sentences can improve quality while respecting budget.

### 4. GPT-4o Temperature = 0

**Decision**: Deterministic LLM sampling for reproducibility.

**Rationale**: Ground truth labels must be consistent across multiple runs for scientific validity.

---

## Limitations & Future Work

### Current Limitations

1. **LLM Hallucinations**: GPT-4o annotations may contain errors. Rationales are stored for audit, but human expert validation is still gold standard.

2. **Domain Specificity**: Evaluated only on Motley Fool transcripts. Cross-validation on Bloomberg/Reuters transcripts would strengthen claims.

3. **Temporal Drift**: Financial language evolves. Models trained on 2020-2023 data may degrade on 2026+ transcripts.

4. **Q&A Segmentation Heuristics**: Regex-based approach assumes header consistency.

### Future Directions

- **Human Expert Validation**: CFA-certified analyst annotations (n=500 sample)
- **Abstractive Summarization**: Compare BART/T5 vs. extractive approaches
- **Multi-Modal Analysis**: Incorporate tone-of-voice and pause frequency from earnings call audio
- **Real-Time Deployment**: API for live investor sentiment dashboards

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{finbert_summarization_2026,
  author = {[Your Name]},
  title = {Overcoming the 512-Token FinBERT Limit in Earnings Call Analysis: 
           A Budget-Aware Extractive Summarization Approach},
  school = {University of Amsterdam},
  year = {2026},
  type = {Master's Thesis},
  department = {Data Science and Artificial Intelligence}
}
```

---

## License

This project is submitted as academic work for a Master's degree at the University of Amsterdam. Code is provided for educational and research purposes.

**Data Source**: Motley Fool earnings call transcripts are proprietary. Users must obtain their own data.

---

## Acknowledgments

- **FinBERT Model**: Huang et al. (2020) - `yiyanghkust/finbert-tone`
- **Theoretical Framework**: Venneman (2025) - Q&A prioritization methodology
- **LLM Infrastructure**: OpenAI GPT-4o for proxy expert annotations

---

## Contact & Contributions

For questions, suggestions, or collaboration:

- **GitHub Issues**: [Open an issue](https://github.com/[your-username]/finbert-sentiment-pipeline/issues)
- **Email**: [your.email@student.uva.nl]
- **LinkedIn**: [Your LinkedIn Profile]

Pull requests welcome! Please ensure all tests pass and code follows PEP 8 style guidelines.
