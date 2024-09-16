import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load the TSV file
file_path = 'datasets/paws-x-test.tsv'  # Replace with your file path
data = pd.read_csv(file_path, sep='\t')

# Define the list of prefixes
prefixes = ['LLama3 zero-shot (Paraph)', 'LLama3 ICL_4 (Paraph)', 'LLama3 zero-shot (Ex. Same Content)','LLama3 ICL_4 (Ex. Same Content)', 'LLama3 zero-shot (Sem Equiv)', 'LLama3 ICL_4 (Sem Equiv)', 'XLM-RoBERTa-EN-ALLTHREE', 'XLM-RoBERTa-EN-PARAPH', 'XLM-RoBERTa-EN-SAMECONTENT', 'XLM-RoBERTa-EN-SEMEQUIV', 'XLM-RoBERTa-EN-ORIG', 'XLM-RoBERTa-EN-EASYNEG-25', 'XLM-RoBERTa-EN-EASYNEG-50', 'XLM-RoBERTa-EN-EASYNEG-75'] 

# Identify columns that match any of the given prefixes
prefix_columns_dict = {prefix: [col for col in data.columns if col.startswith(prefix)] for prefix in prefixes}

languages = data['language'].unique()

# Prepare a DataFrame to store average results
average_results_df = pd.DataFrame(columns=['Language', 'Prefix', 'Average Accuracy', 'Average F1 Score'])

# Function to calculate average metrics for columns with a given prefix
def calculate_average_metrics(data, prefix_columns):
    # Filter columns based on prefix
    filtered_columns = prefix_columns
    if filtered_columns:
        accuracies = []
        f1_scores = []

        for column in filtered_columns:
            labels = data['label']
            preds = data[column]
            accuracy = accuracy_score(labels, preds)
            f1_score = precision_recall_fscore_support(labels, preds, average='macro')[2]

            accuracies.append(accuracy)
            f1_scores.append(f1_score)

        # Calculate average accuracy and F1 score
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_f1_score = sum(f1_scores) / len(f1_scores)

        return avg_accuracy, avg_f1_score
    return None, None

# Calculate average metrics for each language
for language in list(languages) + ['Overall']:
    if language != 'Overall':
        lang_data = data[data['language'] == language]
    else:
        lang_data = data  # For overall metrics, use the whole dataset

    # Calculate metrics for each prefix
    for prefix, prefix_columns in prefix_columns_dict.items():
        avg_accuracy, avg_f1_score = calculate_average_metrics(lang_data, prefix_columns)
        if avg_accuracy is not None:
            # Append the result to the DataFrame using loc
            average_results_df.loc[len(average_results_df)] = [language, prefix, avg_accuracy, avg_f1_score]

# **Custom language order**
custom_language_order = ['en', 'de', 'es', 'fr', 'zh', 'ja', 'ko', 'Overall']

# **Convert the Language column to a categorical type with the specified order**
average_results_df['Language'] = pd.Categorical(average_results_df['Language'], categories=custom_language_order, ordered=True)

# **Sort the DataFrame by the 'Prefix' column and then by the custom order of 'Language'**
average_results_df = average_results_df.sort_values(by=['Prefix', 'Language']).reset_index(drop=True)

# **Round numerical results to 2 decimals**
average_results_df[['Average Accuracy', 'Average F1 Score']] = average_results_df[['Average Accuracy', 'Average F1 Score']].round(4)

# Save the average results to a CSV file
average_results_file_path = 'average_performance_results_per_language.csv'
average_results_df.to_csv(average_results_file_path, index=False)

print(f"Results saved to {average_results_file_path}")
