import pandas as pd
import numpy as np

# List of CSV file paths for non-multilingual datasets
csv_files = ["datasets/onestop_parallel_all_pairs.tsv"]

# Prefix to process for each file
prefix_to_process = {
    "datasets/onestop_parallel_all_pairs.tsv": 'LLama3 ICL_4 (Ex. Same Content)',
}

# Initialize lists to store data for percentage calculations and other metrics
column_data = []
total_columns = 0
# Process each CSV file
for file in csv_files:
    df = pd.read_csv(file, sep="\t")

    # Identify columns that start with the given prefix
    prefix = prefix_to_process[file]
    columns_to_process = [col for col in df.columns if col.startswith(prefix)]
    total_columns = total_columns + len(columns_to_process)

    # Process each identified column
    for column in columns_to_process:
        # Calculate count and sum for the column
        column_count = df[column].count()
        column_sum = df[column].sum()
        # Calculate the percentage of 1s for the column
        if column_count > 0:
            percentage = (column_sum / column_count) * 100
            column_data.append({
                "Column": column,
                "Sample Count": column_count,
                "Count of 1s": column_sum,
                "Percentage of 1s": percentage
            })

# Convert the list to a DataFrame for easier manipulation
column_df = pd.DataFrame(column_data)

# Calculate the overall average percentage across all columns
overall_average_percentage = round(column_df["Percentage of 1s"].mean(), 3) if not column_df.empty else None

# Calculate the standard deviation of the percentages across all columns
overall_std_dev_percentage = round(column_df["Percentage of 1s"].std(), 3) if not column_df.empty else None

# Add overall summary row to the DataFrame
overall_summary = pd.DataFrame({
    "Column": ["Overall"],
    "Sample Count": [column_df["Sample Count"].sum() / total_columns],
    "Count of 1s": [column_df["Count of 1s"].sum() / total_columns],
    "Percentage of 1s": [overall_average_percentage],
    "Standard Deviation of Percentages": [overall_std_dev_percentage]
})

# Append the summary to the original data
result_df = pd.concat([column_df, overall_summary], ignore_index=True)

# Save the results to a CSV file
output_file = "combined_average_stddev_percentages_across_columns.csv"
result_df.to_csv(output_file, index=False)
print(f"CSV file '{output_file}' created successfully.")
