import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("stsbenchmark_train.tsv", sep="\t")

llm_zs_columns = ['LLama3 zero-shot (Paraph)', 'LLama3 zero-shot (Ex. Same Content)', 'LLama3 zero-shot (Sem Equiv)']

# Define the score column and create bins with an interval of 0.2
score_column = 'score'  # Adjust this to your actual score column name
bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.99, 5.01]
df['score_bin'] = pd.cut(df[score_column], bins, right=False)

# Calculate the percentage of '1's in each bin for each technique
techniques = llm_zs_columns + ['Zero_Shot_ALL_THREE'] + ['Zero_Shot_Majority_vote']

percentage_df = df.groupby('score_bin')[techniques].apply(lambda x: (x == 1).mean() * 100).reset_index()

# Plotting
plt.figure(figsize=(12, 8))

for technique in techniques:
    plt.plot(percentage_df['score_bin'].astype(str), percentage_df[technique], marker='o', label=technique)

plt.xlabel('Score Intervals')
plt.ylabel('Percentage of Paraphrases')
plt.title('STSBENCHMARK Training Set Paraphrases Predicted In Bins (Zero Shot)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plotting
plt.figure(figsize=(12, 8))