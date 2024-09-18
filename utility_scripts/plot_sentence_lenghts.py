import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk import download
from Levenshtein import (
    ratio as levenshtein_similarity,
)  # Levenshtein similarity measure
import spacy
from paraphrase_metrics import metrics as pm

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Updated dictionary with actual column names for each file
file_columns_dict = {
    "amr_true_paraphrases.tsv": "AMR Guidelines",
    "stannlp-snli-pre-hyp.tsv": "SNLI",
    "stsbenchmark-test-sts.tsv": "STSBenchmark",
    "fb-anli-pre-hyp.tsv": "ANLI",
    "paws-x-test.tsv": "PAWS-X",
    "onestop_parallel_all_pairs.tsv": "Onestop Parallel",
    "sickr-sts.tsv": "SICK-R",
    "fb-xnli-pre-hyp.tsv": "XNLI",
    "ms-mrpc.tsv": "MRPC",
}

# Custom order for sorting the files
custom_order = [
    "paws-x-test.tsv",
    "stsbenchmark-test-sts.tsv",
    "ms-mrpc.tsv",
    "sickr-sts.tsv",
    "onestop_parallel_all_pairs.tsv",
    "amr_true_paraphrases.tsv",
    "fb-xnli-pre-hyp.tsv",
    "stannlp-snli-pre-hyp.tsv",
    "fb-anli-pre-hyp.tsv",
]


# Function to calculate the average sentence length in tokens
def average_sentence_length(sentences):
    return sentences.str.split().apply(len).mean()


# Function to calculate the average WPD and LD scores for English sentences
def average_wpd_ld(sentence_pairs):
    wpd_scores = []
    ld_scores = []
    for sent1, sent2 in sentence_pairs:
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)
        wpd_score = pm.wpd(doc1, doc2)
        ld_score = pm.ld(doc1, doc2)
        wpd_scores.append(wpd_score)
        ld_scores.append(ld_score)
    avg_wpd = sum(wpd_scores) / len(wpd_scores) if wpd_scores else 0
    avg_ld = sum(ld_scores) / len(ld_scores) if ld_scores else 0
    return avg_wpd, avg_ld


# Get a list of all TSV files in the current directory and filter them based on the dictionary keys
tsv_files = [
    f for f in custom_order if f in file_columns_dict
]  # Sort files based on custom order

# Dictionaries to store the average lengths and similarity measures
average_lengths_normal = {}
average_lengths_large_diff = {}
similarity_normal = {}

# Iterate over each filtered TSV file
for file in tsv_files:
    df = pd.read_csv(file, sep="\t")

    # Calculate average sentence lengths for all sentences
    avg_len_sentence1 = average_sentence_length(df["sentence1"])
    avg_len_sentence2 = average_sentence_length(df["sentence2"])

    # Filter for English sentences in multilingual files if it exists for WPD and LD calculations
    if file == "fb-xnli-pre-hyp.tsv":
        df_en = df[df["lang"] == "en"]  # Filter rows where 'en' column is not NaN
        avg_wpd, avg_ld = average_wpd_ld(zip(df_en["sentence1"], df_en["sentence2"]))
    elif file == "paws-x-test.tsv":
        df_en = df[df["language"] == "en"]  # Filter rows where 'en' column is not NaN
        avg_wpd, avg_ld = average_wpd_ld(zip(df_en["sentence1"], df_en["sentence2"]))
    else:
        avg_wpd, avg_ld = average_wpd_ld(zip(df["sentence1"], df["sentence2"]))

    # Calculate the difference in length
    length_difference = abs(avg_len_sentence1 - avg_len_sentence2)

    # Get the display name from the dictionary
    display_name = file_columns_dict[file]

    # Store the results based on the difference
    if length_difference >= 5:
        average_lengths_large_diff[display_name] = {
            "Premise (Avg Length)": avg_len_sentence1,
            "Hypothesis (Avg Length)": avg_len_sentence2,
        }
    else:
        average_lengths_normal[display_name] = {
            "Sentence1 (Avg Length)": avg_len_sentence1,
            "Sentence2 (Avg Length)": avg_len_sentence2,
        }
        similarity_normal[display_name] = {
            "WPD": avg_wpd,  # Store WPD scores
            "LD": avg_ld,  # Store LD scores
        }
# Create a single figure with 2 subplots (only for sentence lengths)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# Increase spacing between subplots for better readability
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plotting the results for files without large differences (Symmetric)
if average_lengths_normal:
    average_lengths_normal_df = pd.DataFrame(average_lengths_normal).T
    average_lengths_normal_df.plot(kind="bar", ax=ax1, legend=True)  # Set legend to True
    ax1.set_ylabel("Average Length (Tokens)")
    ax1.set_xticklabels(
        average_lengths_normal_df.index, rotation=45, ha="right", fontsize=10
    )  # Add file names to the bottom
    ax1.legend(title='Sentences')  # Optional: Add legend title

# Plotting the results for files with large differences (Asymmetric)
if average_lengths_large_diff:
    average_lengths_large_diff_df = pd.DataFrame(average_lengths_large_diff).T
    average_lengths_large_diff_df.plot(kind="bar", ax=ax2, legend=True)  # Set legend to True
    ax2.set_ylim(0, 55)  # Set maximum y-axis limit to 55
    ax2.set_xticklabels(
        average_lengths_large_diff_df.index, rotation=45, ha="right", fontsize=10
    )  # Add file names to the bottom
    ax2.legend(title='Sentences')  # Optional: Add legend title

# Tight layout for the figures
plt.tight_layout()
plt.savefig("sentence_lengths_combined.pdf")
plt.show()

# Create a separate plot for WPD and LD scores of the symmetric task
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the WPD and LD Scores for files without large differences (Symmetric)
if similarity_normal:
    similarity_normal_df = pd.DataFrame(similarity_normal).T
    similarity_normal_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Scores")
    ax.set_xticklabels(
        similarity_normal_df.index, rotation=45, ha="right", fontsize=10
    )  # Add file names to the bottom

# Tight layout for the WPD and LD score plot
plt.tight_layout()
plt.savefig("wpd_ld_scores_symmetric.pdf")
plt.show()
