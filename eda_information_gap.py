"""
EDA: Do earnings call transcripts even fit in FinBERT's 512-token limit?

Investigating if naive truncation loses the Q&A section, which is supposedly
where the real sentiment signal lives (spontaneous answers vs. scripted remarks).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

DATA_PATH = Path("data/motley-fool-data.pkl")
TOKEN_CAP = 512  # BERT-base limit, can't change this

# Using 1.3x multiplier for word→token conversion
# Probably conservative but better than underestimating and hitting the limit
WORD_TOKEN_RATIO = 1.3

viz_dir = Path("eda_outputs")
viz_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EDA: Token Limit Problem")
print("=" * 80)
print(f"Data: {DATA_PATH}")
print(f"Token cap: {TOKEN_CAP}")
print()

# Load data
transcripts_raw = pd.read_pickle(DATA_PATH)

print(f"Shape: {transcripts_raw.shape}")
print(f"\nColumns: {list(transcripts_raw.columns)}")
print(f"\nMissing values:\n{transcripts_raw.isnull().sum()}")

# Find the text column - different datasets name it differently
text_col = None
for candidate in ['transcript', 'text', 'content', 'body']:
    if candidate in transcripts_raw.columns:
        text_col = candidate
        break

if text_col is None:
    obj_cols = transcripts_raw.select_dtypes(include=['object']).columns
    text_col = obj_cols[0] if len(obj_cols) > 0 else None
    print(f"⚠ Guessing text column: '{text_col}'")

# Quick sanity check
print(f"\nSample (first 500 chars):")
print(transcripts_raw.iloc[0][text_col][:500])
print("...\n")

# Segment transcripts into Remarks vs Q&A
# Need to see if Q&A starts after token 512 - if so, FinBERT won't see it

def split_transcript(txt):
    """Split on Q&A header. Returns (remarks, qa_section)."""
    if not isinstance(txt, str) or len(txt.strip()) == 0:
        return "", ""
    
    # Tried several patterns, these cover most Motley Fool transcripts
    qa_patterns = [
        r'(?:Questions?\s+(?:and|&)\s+Answers?)',
        r'(?:Q\s*&\s*A)',
        r'(?:^Questions?\s+Session)',
    ]
    
    qa_match = None
    for pattern in qa_patterns:
        qa_match = re.search(pattern, txt, re.IGNORECASE | re.MULTILINE)
        if qa_match:
            break
    
    if qa_match:
        split_pos = qa_match.start()
        remarks = txt[:split_pos].strip()
        qa_section = txt[split_pos:].strip()
    else:
        remarks = txt.strip()
        qa_section = ""
    
    return remarks, qa_section

print("Segmenting transcripts...")
transcripts_raw[['remarks', 'q_and_a']] = transcripts_raw[text_col].apply(
    lambda x: pd.Series(split_transcript(x))
)

found_qa = (transcripts_raw['q_and_a'].str.len() > 0).sum()
print(f"Found Q&A in {found_qa}/{len(transcripts_raw)} transcripts")

# Calculate token counts
def word_count(txt):
    if not isinstance(txt, str):
        return 0
    return len(txt.split())

def tokens_from_words(n_words):
    return int(np.ceil(n_words * WORD_TOKEN_RATIO))

transcripts_raw['remarks_words'] = transcripts_raw['remarks'].apply(word_count)
transcripts_raw['qa_words'] = transcripts_raw['q_and_a'].apply(word_count)
transcripts_raw['total_words'] = transcripts_raw[text_col].apply(word_count)

transcripts_raw['remarks_tokens'] = transcripts_raw['remarks_words'].apply(tokens_from_words)
transcripts_raw['qa_tokens'] = transcripts_raw['qa_words'].apply(tokens_from_words)
transcripts_raw['total_tokens'] = transcripts_raw['total_words'].apply(tokens_from_words)

# Where does Q&A start in token space?
transcripts_raw['qa_start_token'] = transcripts_raw['remarks_tokens'] + 1

# Flag: Q&A truncated if it starts past 512
transcripts_raw['qa_truncated'] = transcripts_raw['qa_start_token'] > TOKEN_CAP

# What % of Q&A actually fits?
transcripts_raw['qa_space'] = (TOKEN_CAP - transcripts_raw['remarks_tokens']).clip(lower=0)
transcripts_raw['qa_captured_pct'] = (
    transcripts_raw['qa_space'] / transcripts_raw['qa_tokens']
).replace([np.inf, -np.inf], 0).clip(0, 1)

print(f"\nToken stats:")
print(transcripts_raw['total_tokens'].describe())

truncated_samples = transcripts_raw['qa_truncated'].sum()
print(f"\nQ&A starts after 512 tokens: {truncated_samples} ({truncated_samples/len(transcripts_raw)*100:.1f}%)")

has_qa = transcripts_raw[transcripts_raw['qa_words'] > 0]
if len(has_qa) > 0:
    avg_captured = has_qa['qa_captured_pct'].mean() * 100
    print(f"Avg Q&A captured: {avg_captured:.1f}%")

# Spot check a few examples to verify the split looks right
print(f"\nManual check (3 samples):")

for i, idx in enumerate([0, len(transcripts_raw)//2, len(transcripts_raw)-1], 1):
    row = transcripts_raw.iloc[idx]
    print(f"\n--- Sample {i} ---")
    print(f"Remarks: {row['remarks_tokens']} tokens")
    print(f"Q&A: {row['qa_tokens']} tokens (starts at {row['qa_start_token']})")
    print(f"Truncated: {row['qa_truncated']}")
    if row['qa_words'] > 0:
        print(f"Preview: {row['q_and_a'][:100]}...")
    else:
        print("(No Q&A found)")

# Generate plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Information Gap: 512-Token Limit vs. Earnings Calls", fontsize=14, fontweight='bold')

# Total transcript length distribution
ax1 = axes[0, 0]
ax1.hist(transcripts_raw['total_tokens'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(TOKEN_CAP, color='red', linestyle='--', linewidth=2, label=f'{TOKEN_CAP} limit')
ax1.set_xlabel('Total Tokens')
ax1.set_ylabel('Count')
ax1.set_title('Transcript Length Distribution')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Q&A start positions
ax2 = axes[0, 1]
qa_subset = transcripts_raw[transcripts_raw['qa_words'] > 0]
ax2.hist(qa_subset['qa_start_token'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(TOKEN_CAP, color='red', linestyle='--', linewidth=2, label=f'{TOKEN_CAP} limit')
ax2.set_xlabel('Q&A Start Position (tokens)')
ax2.set_ylabel('Count')
ax2.set_title('Where Q&A Begins')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Truncation breakdown
ax3 = axes[1, 0]
n_truncated = transcripts_raw['qa_truncated'].sum()
n_ok = (~transcripts_raw['qa_truncated']).sum()
n_no_qa = (transcripts_raw['qa_words'] == 0).sum()
counts = [n_truncated, n_ok, n_no_qa]
labels = [
    f'Q&A Truncated\n({n_truncated}, {n_truncated/len(transcripts_raw)*100:.0f}%)',
    f'Q&A OK\n({n_ok}, {n_ok/len(transcripts_raw)*100:.0f}%)',
    f'No Q&A\n({n_no_qa}, {n_no_qa/len(transcripts_raw)*100:.0f}%)'
]
ax3.pie(counts, labels=labels, colors=['#FF6B6B', '#4ECDC4', '#95A5A6'], startangle=90)
ax3.set_title('Truncation Status')

# Section length comparison
ax4 = axes[1, 1]
box_data = [qa_subset['remarks_tokens'], qa_subset['qa_tokens']]
bp = ax4.boxplot(box_data, labels=['Remarks', 'Q&A'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax4.axhline(TOKEN_CAP, color='red', linestyle='--', linewidth=2, label=f'{TOKEN_CAP} limit')
ax4.set_ylabel('Tokens')
ax4.set_title('Remarks vs Q&A Length')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / 'information_gap_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {viz_dir / 'information_gap_analysis.png'}")
plt.close()

# Summary
print(f"\n{'='*60}")
print("Summary")
print('='*60)
print(f"Total: {len(transcripts_raw):,} transcripts")
print(f"Avg tokens: {transcripts_raw['total_tokens'].mean():.0f}")

qa_loss_metrics = transcripts_raw[transcripts_raw['qa_words'] > 0]
if len(qa_loss_metrics) > 0:
    totally_cut = qa_loss_metrics['qa_truncated'].sum()
    print(f"\nOf {len(qa_loss_metrics):,} with Q&A:")
    print(f"  {totally_cut:,} ({totally_cut/len(qa_loss_metrics)*100:.0f}%) - Q&A completely truncated")
    print(f"  Avg captured: {qa_loss_metrics['qa_captured_pct'].mean()*100:.0f}%")

print(f"\nConclusion: Can't just truncate at 512. Need summarization or sliding window.")

# Export
export_cols = ['remarks_tokens', 'qa_tokens', 'total_tokens', 'qa_start_token', 'qa_truncated', 'qa_captured_pct']
transcripts_raw[export_cols].to_csv(viz_dir / 'token_analysis.csv', index=True)
print(f"\nSaved: {viz_dir / 'token_analysis.csv'}")
print(f"\nDone. Outputs in {viz_dir.absolute()}\n")
