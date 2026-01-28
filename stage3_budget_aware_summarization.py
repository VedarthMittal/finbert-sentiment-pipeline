# Budget-aware extractive summarization with strict token limits
# Strategy: prioritize Q&A over prepared remarks (Q&A has more predictive value)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = Path("preprocessing_outputs/transcripts_sentences_only.pkl")
OUTPUT_DIR = Path("stage3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

TOTAL_BUDGET = 450
REMARKS_BUDGET = 150
QA_BUDGET = 300
TOKEN_LIMIT = 512
TOKEN_RATIO = 1.3

FINANCIAL_KEYWORDS = [
    'guidance', 'outlook', 'risk', 'uncertainty', 'volatility', 
    'growth', 'ebitda', 'revenue', 'margin', 'forecast'
]
KEYWORD_BOOST_FACTOR = 1.2

print(f"Loading transcripts from {INPUT_PATH}")
print(f"Token budget: {TOTAL_BUDGET} ({REMARKS_BUDGET} remarks + {QA_BUDGET} Q&A)")
print(f"Keyword boost: {KEYWORD_BOOST_FACTOR}x for {len(FINANCIAL_KEYWORDS)} terms\n")

def estimate_tokens(text, ratio=TOKEN_RATIO):
    # Conservative word-to-token estimate (BERT subword splits)
    if not text or not isinstance(text, str):
        return 0
    word_count = len(text.split())
    return int(np.ceil(word_count * ratio))


def calculate_tfidf_scores(sentences):
    # TF-IDF scores for sentence ranking
    if not sentences or len(sentences) == 0:
        return np.array([])
    
    clean_sentences = [s for s in sentences if s and str(s).strip()]
    if len(clean_sentences) == 0:
        return np.array([])
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words='english', 
            min_df=1,
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        if sentence_scores.max() > 0:
            sentence_scores = sentence_scores / sentence_scores.max()
        
        return sentence_scores
    
    except Exception as e:
        return np.ones(len(clean_sentences))


def calculate_textrank_scores(sentences):
    # PageRank-based sentence centrality scoring
    if not sentences or len(sentences) == 0:
        return np.array([])
    
    clean_sentences = [s for s in sentences if s and str(s).strip()]
    if len(clean_sentences) <= 1:
        return np.ones(len(clean_sentences))
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words='english', 
            min_df=1,
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
        
        if tfidf_matrix.shape[1] == 0:
            return np.ones(len(clean_sentences))
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        graph = nx.from_numpy_array(similarity_matrix)
        pagerank_scores = nx.pagerank(graph, weight='weight', max_iter=100)
        
        scores = np.array([pagerank_scores[i] for i in range(len(clean_sentences))])
        
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    except Exception as e:
        return calculate_tfidf_scores(clean_sentences)


def apply_keyword_boost(sentences, scores, keywords, boost_factor=KEYWORD_BOOST_FACTOR):
    # Boost sentences with financial keywords
    boosted_scores = scores.copy()
    
    for i, sent in enumerate(sentences):
        if not sent or not isinstance(sent, str):
            continue
        
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in keywords):
            boosted_scores[i] *= boost_factor
    
    return boosted_scores


def rank_sentences(sentences, keywords):
    # Combine TF-IDF + TextRank + keyword boost
    if not sentences or len(sentences) == 0:
        return []
    
    clean_sentences = [s for s in sentences if s and str(s).strip()]
    if len(clean_sentences) == 0:
        return []
    
    tfidf_scores = calculate_tfidf_scores(clean_sentences)
    textrank_scores = calculate_textrank_scores(clean_sentences)
    
    if len(tfidf_scores) != len(clean_sentences):
        tfidf_scores = np.ones(len(clean_sentences))
    if len(textrank_scores) != len(clean_sentences):
        textrank_scores = np.ones(len(clean_sentences))
    
    combined_scores = (tfidf_scores + textrank_scores) / 2
    boosted_scores = apply_keyword_boost(clean_sentences, combined_scores, keywords)
    
    ranked = [(i, sent, score) for i, (sent, score) in enumerate(zip(clean_sentences, boosted_scores))]
    ranked.sort(key=lambda x: x[2], reverse=True)
    
    return ranked


def select_sentences_by_budget(ranked_sentences, token_budget, ratio=TOKEN_RATIO):
    # Greedy selection: keep adding highest-scored sentences until budget exhausted
    selected = []
    cumulative_tokens = 0
    
    for idx, sent, score in ranked_sentences:
        sent_tokens = estimate_tokens(sent, ratio=ratio)
        
        if cumulative_tokens + sent_tokens <= token_budget:
            selected.append((idx, sent))
            cumulative_tokens += sent_tokens
    
    selected.sort(key=lambda x: x[0])
    selected_sentences = [sent for idx, sent in selected]
    leftover_budget = token_budget - cumulative_tokens
    
    return selected_sentences, cumulative_tokens, leftover_budget


def generate_budget_aware_summary(remarks_sents, qa_sents, 
                                   remarks_budget=REMARKS_BUDGET, 
                                   qa_budget=QA_BUDGET,
                                   keywords=FINANCIAL_KEYWORDS):
    # Budget-aware summarization with rollover from remarks to Q&A
    summary_parts = []
    remarks_tokens_used = 0
    qa_tokens_used = 0
    remarks_count = 0
    qa_count = 0
    
    if remarks_sents and len(remarks_sents) > 0:
        ranked_remarks = rank_sentences(remarks_sents, keywords)
        selected_remarks, remarks_tokens_used, leftover_remarks = select_sentences_by_budget(
            ranked_remarks, remarks_budget, TOKEN_RATIO
        )
        remarks_count = len(selected_remarks)
        summary_parts.extend(selected_remarks)
    else:
        leftover_remarks = remarks_budget
    
    if qa_sents and len(qa_sents) > 0:
        expanded_qa_budget = qa_budget + leftover_remarks
        ranked_qa = rank_sentences(qa_sents, keywords)
        selected_qa, qa_tokens_used, leftover_qa = select_sentences_by_budget(
            ranked_qa, expanded_qa_budget, TOKEN_RATIO
        )
        qa_count = len(selected_qa)
        summary_parts.extend(selected_qa)
    
    summary_text = " ".join(summary_parts)
    total_tokens = remarks_tokens_used + qa_tokens_used
    
    return {
        'summary_text': summary_text,
        'remarks_tokens': remarks_tokens_used,
        'qa_tokens': qa_tokens_used,
        'total_tokens': total_tokens,
        'remarks_count': remarks_count,
        'qa_count': qa_count,
        'total_count': remarks_count + qa_count
    }


print(f"Loading: {INPUT_PATH}")
transcripts_df = pd.read_pickle(INPUT_PATH)
print(f"Loaded {len(transcripts_df):,} transcripts\n")

print("Generating summaries...")
results = []

for idx, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df), desc="Progress", unit=" transcript"):
    remarks_sents = row['remarks_sentences'] if isinstance(row['remarks_sentences'], list) else []
    qa_sents = row['qa_sentences'] if isinstance(row['qa_sentences'], list) else []
    
    summary_result = generate_budget_aware_summary(
        remarks_sents=remarks_sents,
        qa_sents=qa_sents,
        remarks_budget=REMARKS_BUDGET,
        qa_budget=QA_BUDGET,
        keywords=FINANCIAL_KEYWORDS
    )
    
    final_token_count = estimate_tokens(summary_result['summary_text'], TOKEN_RATIO)
    
    results.append({
        'ticker': row['ticker'],
        'date': row['date'],
        'n_remarks_sentences': row.get('n_remarks_sentences', 0),
        'n_qa_sentences': row.get('n_qa_sentences', 0),
        'n_total_sentences': row.get('n_total_sentences', 0),
        'extracted_remarks': summary_result['remarks_count'],
        'extracted_qa': summary_result['qa_count'],
        'extracted_total': summary_result['total_count'],
        'budgeted_tokens': summary_result['total_tokens'],
        'summary_tokens': final_token_count,
        'fits_512_limit': final_token_count <= TOKEN_LIMIT,
        'extractive_summary': summary_result['summary_text']
    })

summaries_df = pd.DataFrame(results)
print(f"\nGenerated {len(summaries_df):,} summaries")

# Stats
mean_tokens = summaries_df['summary_tokens'].mean()
median_tokens = summaries_df['summary_tokens'].median()
max_tokens = summaries_df['summary_tokens'].max()
min_tokens = summaries_df['summary_tokens'].min()

print(f"\nToken stats: mean={mean_tokens:.1f}, median={median_tokens:.1f}, max={max_tokens}, min={min_tokens}")

compliant_count = summaries_df['fits_512_limit'].sum()
compliant_pct = (compliant_count / len(summaries_df)) * 100
print(f"Compliance: {compliant_count:,}/{len(summaries_df):,} ({compliant_pct:.2f}%)")

if compliant_pct < 100.0:
    exceeding = summaries_df[~summaries_df['fits_512_limit']]
    print(f"Warning: {len(exceeding)} summaries exceed limit")
    if len(exceeding) > 0:
        print(exceeding[['ticker', 'date', 'summary_tokens']].head())

# Export
output_pkl = OUTPUT_DIR / "extractive_summaries.pkl"
summaries_df.to_pickle(output_pkl)
print(f"\nSaved: {output_pkl}")

stats_csv = OUTPUT_DIR / "summarization_statistics.csv"
summaries_df.to_csv(stats_csv, index=False)
print(f"Saved: {stats_csv}")

sample_txt = OUTPUT_DIR / "sample_summaries.txt"
with open(sample_txt, 'w', encoding='utf-8') as f:
    f.write("Budget-Aware Extractive Summaries - Samples\n")
    f.write("=" * 80 + "\n\n")
    
    for idx in range(min(5, len(summaries_df))):
        row = summaries_df.iloc[idx]
        f.write(f"\nTranscript {idx + 1}: {row['ticker']} ({row['date']})\n")
        f.write(f"Original: {row['n_total_sentences']} sentences\n")
        f.write(f"Extracted: {row['extracted_total']} sentences ({row['extracted_remarks']} remarks + {row['extracted_qa']} Q&A)\n")
        f.write(f"Tokens: {row['summary_tokens']} / {TOKEN_LIMIT}\n\n")
        f.write("Summary:\n")
        f.write(row['extractive_summary'][:1000])
        if len(row['extractive_summary']) > 1000:
            f.write(f"\n... [truncated]\n")
        f.write("\n" + "-" * 80 + "\n")

print(f"Saved: {sample_txt}")

print(f"\nDone. {len(summaries_df):,} summaries generated, compliance={compliant_pct:.1f}%")
print(f"Ready for Stage 4 (FinBERT sentiment analysis).\n")
