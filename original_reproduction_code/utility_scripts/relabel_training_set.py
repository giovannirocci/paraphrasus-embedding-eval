import os
import pandas as pd
from datasets import load_dataset

# Define the folder to save the files
dataset_file = 'relabeled_train_paws_x_en.tsv'
df = pd.read_csv(dataset_file, sep="\t")
llm_zs_columns = ['LLama3 zero-shot (Paraph)', 'LLama3 zero-shot (Ex. Same Content)', 'LLama3 zero-shot (Sem Equiv)']

# Zero-Shot Majority Vote
df['Zero_Shot_Majority_vote'] = df[llm_zs_columns].mode(axis=1)[0]

# Zero-Shot Paraphrase and Same Content Agreement
df['Zero_Shot_Paraph_AND_Same_Content'] = (
    (df['LLama3 zero-shot (Paraph)'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['LLama3 zero-shot (Paraph)'] == 1)
).astype(int)

# Zero-Shot Paraphrase and Semantic Equivalence Agreement
df['Zero_Shot_Paraph_AND_Sem_Equiv']  = (
    (df['LLama3 zero-shot (Paraph)'] == df['LLama3 zero-shot (Sem Equiv)']) & 
    (df['LLama3 zero-shot (Paraph)'] == 1)
).astype(int)

# Zero-Shot Paraphrase and Semantic Equivalence Agreement
df['Zero_Shot_ALL_THREE']  = (
    (df['Zero_Shot_Paraph_AND_Sem_Equiv'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['Zero_Shot_Paraph_AND_Sem_Equiv'] == 1)
).astype(int)

df.to_csv(dataset_file, sep='\t')