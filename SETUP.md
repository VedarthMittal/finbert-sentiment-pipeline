# Repository Setup Guide

## Quick Start for GitHub Hosting

This guide ensures your repository is ready for public deployment on GitHub **without exposing proprietary data**.

---

## Pre-Deployment Checklist

### 1. Verify Directory Structure

After cloning, your repository should have:

```
Individual Assignment/
├── data/                          # EMPTY (gitignored)
├── outputs/                       # EMPTY (gitignored)
├── eda_information_gap.py         # Stage 1: EDA
├── preprocessing_stage2.py        # Stage 2: Preprocessing
├── stage3_budget_aware_summarization.py  # Stage 3: Summarization
├── stage4_gpu_optimized.py        # Stage 4: Inference
├── stage5_ground_truth_llm.py     # Stage 5: Ground Truth
├── stage6_evaluation_metrics.py   # Stage 6: Evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

**Action Required**: User must manually add `motley-fool-data.pkl` to `data/` directory.

### 2. Install Dependencies

```bash
# Activate your Python 3.13 virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt

# Download NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Set OpenAI API Key (Stage 5 Only)

For ground truth generation via GPT-4o:

```bash
# On Windows (PowerShell):
$env:OPENAI_API_KEY="sk-proj-..."

# On macOS/Linux:
export OPENAI_API_KEY="sk-proj-..."

# Or use a .env file (already gitignored):
echo "OPENAI_API_KEY=sk-proj-..." > .env
```

### 4. Verify CUDA Availability (Stage 4 Optimization)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If False, Stage 4 will use CPU (slower but functional).

---

## Running the Pipeline

Execute stages in order:

```bash
# Stage 1: Information Gap Analysis
python eda_information_gap.py
# Output: eda_outputs/token_statistics.csv

# Stage 2: Sentence Tokenization
python preprocessing_stage2.py
# Output: preprocessing_outputs/transcripts_sentences_only.pkl

# Stage 3: Budget-Aware Summarization
python stage3_budget_aware_summarization.py
# Output: stage3_output/extractive_summaries.pkl

# Stage 4: FinBERT Sentiment Inference
python stage4_gpu_optimized.py
# Output: stage4_output/final_sentiment_results.pkl

# Stage 5: Ground Truth Generation (requires API key)
python stage5_ground_truth_llm.py
# Output: outputs/05_ground_truth_eval.pkl

# Stage 6: Evaluation Metrics & Visualization
python stage6_evaluation_metrics.py
# Output: outputs/06_evaluation_plots/*.png
```

---

## GitHub Deployment Steps

### Initialize Git Repository

```bash
cd "C:\Users\mitta\Desktop\Applied AI\Individual Assignment"

# Initialize repo
git init

# Add all tracked files
git add .

# Commit
git commit -m "Initial commit: FinBERT 512-token limitation research pipeline"
```

### Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `finbert-sentiment-pipeline` (or your choice)
3. Description: "Overcoming the 512-Token FinBERT Limit in Earnings Call Analysis"
4. **Public** or **Private**: Your choice
5. **DO NOT** initialize with README (you already have one)

### Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/[your-username]/finbert-sentiment-pipeline.git

# Push
git branch -M main
git push -u origin main
```

---

## Verification Tests

### Test 1: Data Exclusion

```bash
git status
# Should NOT show data/*.pkl or outputs/*.pkl
```

### Test 2: Path Integrity

```bash
python -c "from pathlib import Path; print(Path('data/motley-fool-data.pkl').exists())"
# Should print: True (if you added the data file)
```

### Test 3: Dependency Resolution

```bash
pip list | grep -E "torch|transformers|openai"
# Should show all installed versions
```

---

## Common Issues

### Issue 1: ModuleNotFoundError for NLTK punkt

**Solution**:
```bash
python -m nltk.downloader punkt punkt_tab
```

### Issue 2: CUDA Out of Memory (Stage 4)

**Solution**: Reduce batch size in `stage4_gpu_optimized.py`:
```python
BATCH_SIZE = 16  # Down from 32
```

### Issue 3: OpenAI API Rate Limit (Stage 5)

**Solution**: The code already includes 2-second delays. If still hitting limits, increase:
```python
time.sleep(5)  # In stage5_ground_truth_llm.py
```

---

## File Size Management

The `.gitignore` excludes:

- `data/` → Raw transcripts (~500MB)
- `outputs/*.pkl` → Intermediate artifacts (~100MB)
- `outputs/*.csv` → Results (~5MB)
- `venv/` → Python packages (~500MB)

**Repository size after push**: ~50KB (code + docs only)

---

## Security Checklist

Before pushing to public GitHub:

- [ ] `.env` file is gitignored
- [ ] OpenAI API key NOT hardcoded in any .py file
- [ ] `data/motley-fool-data.pkl` is gitignored
- [ ] No sensitive company information in code comments
- [ ] `outputs/` directory is empty (or gitignored)

---

## Post-Deployment

After pushing to GitHub, add a **badge** to README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.13+-blue)
![License](https://img.shields.io/badge/license-Academic-green)
```

And update the README with actual results from Stages 4-6.

---

## Contact

For questions about this setup:
- GitHub Issues: https://github.com/[your-username]/finbert-sentiment-pipeline/issues
- Email: [your.email@student.uva.nl]
