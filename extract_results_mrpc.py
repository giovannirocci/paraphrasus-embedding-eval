import pandas as pd
import math
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Load the data from a TSV file
data = pd.read_csv('datasets/ms-mrpc.tsv', sep='\t')

true_labels = data['label']

# List of prefixes to evaluate
prefixes = ['XLM-RoBERTa-EN-EASYNEG-75', 'XLM-RoBERTa-EN-EASYNEG-50', 'XLM-RoBERTa-EN-EASYNEG-25', 'XLM-RoBERTa-EN-ORIG', 'XLM-RoBERTa-EN-PARAPH', 'XLM-RoBERTa-EN-SAMECONTENT', 'XLM-RoBERTa-EN-SEMEQUIV', 'XLM-RoBERTa-EN-ALLTHREE']
llm_zs_columns = ["LLama3 zero-shot (Paraph)", "LLama3 zero-shot (Ex. Same Content)", "LLama3 zero-shot (Sem Equiv)", "LLama3 ICL_4 (Paraph)", "LLama3 ICL_4 (Ex. Same Content)", "LLama3 ICL_4 (Sem Equiv)"] 
prefixes = prefixes + llm_zs_columns

# Initialize an empty DataFrame to store results
results = pd.DataFrame(columns=['Model', 'LabelType', 'Correct Predictions', 'Incorrect Predictions', 'Total True', 'Total Predicted', 'Macro F1', 'Accuracy'])

# Loop through each prefix and calculate metrics
for prefix in prefixes:
    # Find all columns that match the prefix
    model_columns = [col for col in data.columns if col.startswith(prefix)]

    # Initialize variables to accumulate totals across all columns
    total_f1_scores = []
    total_accuracy_scores = []

    # Calculate confusion matrices, macro F1 score, and accuracy for each model column matching the prefix
    for model in model_columns:
        predictions = data[model]
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

        # Calculate the F1 score and accuracy for this column
        f1 = f1_score(true_labels, predictions, average='macro')
        accuracy = accuracy_score(true_labels, predictions)
        
        total_f1_scores.append(f1)
        total_accuracy_scores.append(accuracy)

        # Calculate the true positives and true negatives for this column
        true_positives = sum(true_labels)
        true_negatives = len(true_labels) - true_positives
        predicted_positives = sum(predictions)
        predicted_negatives = len(predictions) - predicted_positives

        # Calculate the correct and incorrect predictions
        correct_predictions_pos = tp
        incorrect_predictions_pos = fp
        correct_predictions_neg = tn
        incorrect_predictions_neg = fn

        # Append results for positives and negatives
        results = results.append({
            'Model': model,
            'LabelType': 'Positives',
            'Correct Predictions': correct_predictions_pos,
            'Incorrect Predictions': incorrect_predictions_pos,
            'Total True': true_positives,
            'Total Predicted': predicted_positives,
            'Macro F1': f1,
            'Accuracy': accuracy
        }, ignore_index=True)

        results = results.append({
            'Model': model,
            'LabelType': 'Negatives',
            'Correct Predictions': correct_predictions_neg,
            'Incorrect Predictions': incorrect_predictions_neg,
            'Total True': true_negatives,
            'Total Predicted': predicted_negatives,
            'Macro F1': f1,
            'Accuracy': accuracy
        }, ignore_index=True)

    # Calculate the average macro F1 score and accuracy across all models for the current prefix
    average_macro_f1 = sum(total_f1_scores) / len(total_f1_scores)
    average_accuracy = sum(total_accuracy_scores) / len(total_accuracy_scores)

    # Compute the variance and standard deviation for F1 scores
    f1_variance = sum((x - average_macro_f1) ** 2 for x in total_f1_scores) / len(total_f1_scores)
    f1_standard_deviation = math.sqrt(f1_variance)

    # Compute the variance and standard deviation for accuracy scores
    accuracy_variance = sum((x - average_accuracy) ** 2 for x in total_accuracy_scores) / len(total_accuracy_scores)
    accuracy_standard_deviation = math.sqrt(accuracy_variance)

    # Append the average macro F1, accuracy, and their standard deviations to the results DataFrame
    results = results.append({
        'Model': prefix,
        'LabelType': 'Average',
        'Correct Predictions': '',
        'Incorrect Predictions': '',
        'Total True': '',
        'Total Predicted': '',
        'Macro F1': average_macro_f1,
        'Accuracy': average_accuracy
    }, ignore_index=True)

    results = results.append({
        'Model': prefix,
        'LabelType': 'Standard Deviation',
        'Correct Predictions': '',
        'Incorrect Predictions': '',
        'Total True': '',
        'Total Predicted': '',
        'Macro F1': f1_standard_deviation,
        'Accuracy': accuracy_standard_deviation
    }, ignore_index=True)

# Save results to a CSV file
results.to_csv('detailed_confusion_matrices_and_metrics_averaged.csv', index=False)

print("CSV file 'detailed_confusion_matrices_and_metrics_averaged.csv' created successfully.")
print(f"Standard Deviation of Macro F1 scores: {f1_standard_deviation}")
print(f"Standard Deviation of Accuracy scores: {accuracy_standard_deviation}")
