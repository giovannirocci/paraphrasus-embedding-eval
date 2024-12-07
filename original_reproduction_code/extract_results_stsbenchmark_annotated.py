import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Load your datasets
df = pd.read_csv("datasets/stsbenchmark-test-sts.tsv", sep="\t")
df_sickr = pd.read_csv("datasets/sickr-sts.tsv", sep="\t")  # Load the new SICK-R dataset

# Define the columns for zero-shot and ICL_4
llm_zs_columns = ['LLama3 zero-shot (Paraph)', 'LLama3 zero-shot (Ex. Same Content)', 'LLama3 zero-shot (Sem Equiv)']
human_annotation_column = 'Human Annotation - Consensus'

# Map the Human Annotation column from 1.0 and 0.0 to 1 and 0
df[human_annotation_column] = df[human_annotation_column].map({1.0: 1, 0.0: 0, -100.0: 0})

# Zero-Shot Majority Vote
df['Zero_Shot_Majority_vote'] = df[llm_zs_columns].mode(axis=1)[0]
df['Zero_Shot_Majority_vote'] = df[llm_zs_columns].mode(axis=1)[0]

# Zero-Shot Paraphrase and Semantic Equivalence Agreement
df['Zero_Shot_ALL_THREE'] = (
    (df['LLama3 zero-shot (Paraph)'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['LLama3 zero-shot (Sem Equiv)'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['LLama3 zero-shot (Ex. Same Content)'] == 1)
).astype(int)

df['Zero_Shot_ALL_THREE'] = (
    (df['LLama3 zero-shot (Paraph)'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['LLama3 zero-shot (Sem Equiv)'] == df['LLama3 zero-shot (Ex. Same Content)']) & 
    (df['LLama3 zero-shot (Ex. Same Content)'] == 1)
).astype(int)

df_easy = df[df['score'] < 3]
df_hard = df[df['score'] >= 4]

# Repeat for the SICK-R dataset
df_sickr[human_annotation_column] = -100
# Map the Human Annotation column from 1.0 and 0.0 to 1 and 0
df_sickr[human_annotation_column] = df_sickr[human_annotation_column].map({1.0: 1, 0.0: 0, -100.0: 0})
df_sickr_easy = df_sickr[df_sickr['score'] < 3]  # Filter for score < 3

# Zero-Shot Majority Vote for SICK-R
df_sickr_easy['Zero_Shot_Majority_vote'] = df_sickr_easy[llm_zs_columns].mode(axis=1)[0]

# Zero-Shot Paraphrase and Semantic Equivalence Agreement for SICK-R
df_sickr_easy['Zero_Shot_ALL_THREE'] = (
    (df_sickr_easy['LLama3 zero-shot (Paraph)'] == df_sickr_easy['LLama3 zero-shot (Ex. Same Content)']) & 
    (df_sickr_easy['LLama3 zero-shot (Sem Equiv)'] == df_sickr_easy['LLama3 zero-shot (Ex. Same Content)']) & 
    (df_sickr_easy['LLama3 zero-shot (Ex. Same Content)'] == 1)
).astype(int)

# Function to calculate metrics for each model in the given dataframe
def evaluate_models(df, dataset_label, easy_eval):
    results = []
    
    # Evaluate each model including llm_zs_columns and prefix-specific models
    for column in df.columns:
        if column in llm_zs_columns or column.startswith("XLM-RoBERTa-EN-") or column in ['Zero_Shot_Majority_vote', 'Zero_Shot_ALL_THREE', 'LLama3 ICL_4 (Paraph)', 'LLama3 ICL_4 (Sem Equiv)', 'LLama3 ICL_4 (Ex. Same Content)']:
            accuracy, f1 = evaluate_column(df, column, human_annotation_column)
            if easy_eval:
                results.append({
                    "Model": column, 
                    f"{dataset_label} Error%": round((100 - accuracy * 100), 2)
                })
            else:
                results.append({
                    "Model": column, 
                    f"{dataset_label} Accuracy": round(accuracy * 100, 2), 
                    f"{dataset_label} F1": round(f1, 2)
                })
    
    # Calculate and return results as a DataFrame
    return pd.DataFrame(results)

# Function to calculate accuracy and F1 score for a given column
def evaluate_column(df, pred_column, true_column):
    accuracy = accuracy_score(df[true_column], df[pred_column])
    f1 = f1_score(df[true_column], df[pred_column], average="macro")
    return accuracy, f1

# List of prefixes to evaluate
prefixes = ['XLM-RoBERTa-EN-ORIG-', 'XLM-RoBERTa-EN-ALLTHREE-', 'XLM-RoBERTa-EN-SAMECONTENT', 'XLM-RoBERTa-EN-PARAPH', 'XLM-RoBERTa-EN-SEMEQUIV', 'XLM-RoBERTa-EN-EASYNEG-25', 'XLM-RoBERTa-EN-EASYNEG-50', 'XLM-RoBERTa-EN-EASYNEG-75']

# Initialize an empty DataFrame to store all results
all_results = pd.DataFrame()

# Include results for llm_zs_columns and combined results
print(f"\nEvaluating Zero-Shot models and combined results (Easy dataset)")
easy_results = evaluate_models(df_easy, "STSB-Easy", easy_eval=True)
print(f"\nEvaluating Zero-Shot models and combined results (Hard dataset)")
hard_results = evaluate_models(df_hard, "STSB-Hard", easy_eval=False)

# Evaluate SICK-R for Zero-Shot models and combined results
print(f"\nEvaluating Zero-Shot models and combined results (SICK-R Easy dataset)")
sickr_easy_results = evaluate_models(df_sickr_easy, "SICKR-Easy", easy_eval=True)

# Merge all results on 'Model' column
merged_results = pd.merge(easy_results, hard_results, on='Model', how='outer')
merged_results = pd.merge(merged_results, sickr_easy_results, on='Model', how='outer')

# Evaluate models for both easy and hard datasets for each prefix
for prefix in prefixes:
    # Filter columns that start with the current prefix, and include the human_annotation_column
    easy_df = df_easy[[human_annotation_column] + [col for col in df_easy.columns if col.startswith(prefix)]]
    hard_df = df_hard[[human_annotation_column] + [col for col in df_hard.columns if col.startswith(prefix)]]
    sickr_easy_df = df_sickr_easy[[human_annotation_column] + [col for col in df_sickr_easy.columns if col.startswith(prefix)]]

    # Evaluate models for the Easy, Hard, and SICK-R Easy datasets
    easy_results = evaluate_models(easy_df, "STSB-Easy", easy_eval=True)
    hard_results = evaluate_models(hard_df, "STSB-Hard", easy_eval=False)
    sickr_easy_results = evaluate_models(sickr_easy_df, "SICKR-Easy", easy_eval=True)

    easy_accuracy_std = np.std(easy_results["STSB-Easy Error%"])
    hard_accuracy_std = np.std(hard_results["STSB-Hard Accuracy"])
    hard_f1_std = np.std(hard_results["STSB-Hard F1"])
    sickr_easy_accuracy_std = np.std(sickr_easy_results["SICKR-Easy Error%"])
    
    # Print the standard deviations
    print(f"Standard Deviation of Error% for {prefix} in STSB-Easy: {easy_accuracy_std:.2f}")
    print(f"Standard Deviation of Accuracy for {prefix} in STSB-Hard: {hard_accuracy_std:.2f}")
    print(f"Standard Deviation of F1 Score for {prefix} in STSB-Hard: {hard_f1_std:.2f}")
    print(f"Standard Deviation of Error% for {prefix} in SICKR-Easy: {sickr_easy_accuracy_std:.2f}")
    
    # Merge the results on 'Model' column
    prefix_results = pd.merge(easy_results, hard_results, on='Model', how='outer')
    prefix_results = pd.merge(prefix_results, sickr_easy_results, on='Model', how='outer')

    # Calculate the average scores for the prefix across datasets
    avg_easy_accuracy = prefix_results["STSB-Easy Error%"].mean()
    avg_hard_accuracy = prefix_results["STSB-Hard Accuracy"].mean()
    avg_hard_f1 = prefix_results["STSB-Hard F1"].mean()
    avg_sickr_easy_accuracy = prefix_results["SICKR-Easy Error%"].mean()

    # Add average row
    avg_row = {
        "Model": f"{prefix} (Average)",
        "STSB-Easy Error%": round(avg_easy_accuracy, 2),
        "STSB-Hard Accuracy": round(avg_hard_accuracy, 2),
        "STSB-Hard F1": round(avg_hard_f1, 2),
        "SICKR-Easy Error%": round(avg_sickr_easy_accuracy, 2)
    }

    # Append average row to merged results
    # prefix_results = prefix_results.append(avg_row, ignore_index=True)
    prefix_results = pd.concat([prefix_results, pd.DataFrame([avg_row])], ignore_index=True)

    # Add to all results
    all_results = pd.concat([all_results, prefix_results], ignore_index=True)

# Combine Zero-Shot results with other prefix-based results
all_results = pd.concat([merged_results, all_results], ignore_index=True).drop_duplicates(subset=['Model'])

# Save results to TSV file
all_results.to_csv("model_performance_results_combined.tsv", sep="\t", index=False)

print("Results saved to 'model_performance_results_combined.tsv'")
