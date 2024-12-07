import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score

# Load your dataset
df = pd.read_csv("stsbenchmark-test-sts.tsv", sep="\t")

# Define the columns for zero-shot and ICL_4
llm_zs_columns = ['LLama3 zero-shot (Paraph)', 'LLama3 zero-shot (Sem Equiv)', 'LLama3 zero-shot (Ex. Same Content)']
human_annotation_column = 'Human Annotation - Consensus'

# Map the Human Annotation column from 1.0 and 0.0 to 1 and 0
df[human_annotation_column] = df[human_annotation_column].map({1.0: 1, 0.0: 0, -100.0: -100})

# Define the score column and create bins with an interval of 0.5
score_column = 'score'
bins = np.arange(0, 5.51, 0.5)  # Create bins from 0 to 5.51 with a step of 0.5
df['score_bin'] = pd.cut(df[score_column], bins, right=False)

# Adjust the label for the last interval
bin_labels = [f'[{b:.1f}, {b + 0.5:.1f})' for b in bins[:-2]] + ['[5.0, 5.0]']
df['score_bin'] = pd.cut(df[score_column], bins=bins, labels=bin_labels, right=False)

# Filter out the unwanted range (-100 values between 1.0 to 4.0)
df_filtered = df

# Calculate the percentage of '1's in each bin for each technique
techniques = llm_zs_columns + ["XLM-RoBERTa-EN-ORIG-V3-1"] + [human_annotation_column]
percentage_df = df_filtered.groupby('score_bin')[techniques].apply(lambda x: (x == 1).mean() * 100).reset_index()

# Custom map for the column names to more readable format
column_name_map = {
    'LLama3 zero-shot (Paraph)': 'LLM Zero-Shot (P1 - Paraphrase)',
    'LLama3 zero-shot (Sem Equiv)': 'LLM Zero-Shot (P2 - Semantic Equivalent)',
    'LLama3 zero-shot (Ex. Same Content)': 'LLM Zero-Shot (P3 - Exact Same Content)',
    'XLM-RoBERTa-EN-ORIG-V3-1': 'XLM-RoBERTa-PAWS-EN',
    'Human Annotation - Consensus': 'Human Annotation'
}

# Plotting
plt.figure(figsize=(12, 8))

for technique in techniques:
    plt.plot(percentage_df['score_bin'].astype(str), percentage_df[technique], marker='o', label=column_name_map[technique])

plt.xlabel('Score Intervals', fontsize=14)
plt.ylabel('Percentage of Paraphrases', fontsize=14)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)

# Set the legend with a larger font size and position it at the top left inside the plot
plt.legend(loc='upper left', fontsize=18)

plt.tight_layout()
plt.savefig("stsbenchmark_curve_custom.pdf")
plt.show()
