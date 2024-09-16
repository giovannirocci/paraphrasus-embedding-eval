import pandas as pd
from collections import defaultdict
import numpy as np

# List of CSV file paths for non-multilingual datasets
csv_files = ["datasets/fb-anli-pre-hyp.tsv", "datasets/fb-anli-hyp-pre.tsv"]

column = 'LLama3 ICL_4 (Ex. Same Content)'
# Prefix to process for each file
prefix_to_process = {
    "datasets/fb-anli-pre-hyp.tsv": column,
    "datasets/fb-anli-hyp-pre.tsv": column
}
total_columns = 0
# Initialize a dictionary to store combined results
combined_results = defaultdict(lambda: defaultdict(lambda: {"count": 0, "sum": 0, "percentages": []}))

# Process each CSV file
for file in csv_files:
    df = pd.read_csv(file, sep="\t")

    # Identify columns that start with the given prefix
    prefix = prefix_to_process[file]
    columns_to_process = [col for col in df.columns if col.startswith(prefix)]
    total_columns = total_columns + len(columns_to_process)
    # Process each identified column
    for column in columns_to_process:
        # Iterate over the two cases: contradiction (-1) and neutral (0)
        for score_value, case_name in [(-1, "contradiction"), (0, "neutral")]:
            # Filter the DataFrame based on the score value
            filtered_df = df[df["score"] == score_value]

            # Calculate required metrics
            total_count = filtered_df[column].count()
            total_sum = filtered_df[column].sum()

            if total_count > 0:
                # Calculate the percentage of 1s
                percentage_of_1s = (total_sum / total_count) * 100
                combined_results[column][case_name]["percentages"].append(percentage_of_1s)

            # Store the results in the combined_results dictionary
            combined_results[column][case_name]["count"] += total_count / len(csv_files)
            combined_results[column][case_name]["sum"] += total_sum / len(csv_files)

# List to store the result DataFrames for each column
result_dfs = []

# Initialize variables to calculate overall averages split by case
overall_counts = {"contradiction": 0, "neutral": 0}
overall_sums = {"contradiction": 0, "neutral": 0}
percentage_data = {"contradiction": [], "neutral": []}
case_counter = {"contradiction": 0, "neutral": 0}
total_columns = total_columns / 2
# Post-process combined results to calculate percentages and format the final output
for column, case_data in combined_results.items():
    for case_name, metrics in case_data.items():
        # Compute the average percentage
        percentages = metrics["percentages"]
        avg_percentage = round(np.mean(percentages), 3) if percentages else None

        # Calculate standard deviation across percentages
        std_dev_percentage = round(np.std(percentages), 3) if percentages else None

        # Update overall counts, sums, and values for each case
        overall_counts[case_name] += metrics["count"]
        overall_sums[case_name] += metrics["sum"]
        percentage_data[case_name].extend(percentages)
        case_counter[case_name] += 1

        # Creating the result DataFrame for each case
        result_df = pd.DataFrame({
            "Metric": [f"{column} ({case_name})"],
            "Sample Count": [metrics["count"]],
            "Count of 1s": [metrics["sum"]],
            "Percentage of 1s": [avg_percentage],
            "Standard Deviation of Percentages": [std_dev_percentage]
        })

        # Append the result DataFrame to the list
        result_dfs.append(result_df)

        # Append an empty DataFrame for the empty line
        result_dfs.append(pd.DataFrame({"Metric": [""], "Sample Count": [""], "Count of 1s": [""], "Percentage of 1s": [""], "Standard Deviation of Percentages": [""]}))

# Calculate overall averages split by case
final_overall_percentages = {case_name: np.mean(percentage_data[case_name]) if percentage_data[case_name] else None for case_name in case_counter}
final_overall_std_devs = {case_name: np.std(percentage_data[case_name]) if percentage_data[case_name] else None for case_name in percentage_data}

# Creating the overall average DataFrames for each case
for case_name in ["contradiction", "neutral"]:
    final_overall_avg_df = pd.DataFrame({
        "Metric": [f"Overall Average ({case_name})"],
        "Sample Count": [round(overall_counts[case_name], 3) /total_columns],
        "Count of 1s": [round(overall_sums[case_name], 3) / total_columns],
        "Percentage of 1s": [round(final_overall_percentages[case_name], 3) if final_overall_percentages[case_name] is not None else None],
        "Standard Deviation of Percentages": [round(final_overall_std_devs[case_name], 3) if final_overall_std_devs[case_name] is not None else None]
    })

    # Append the overall average DataFrame to the results
    result_dfs.append(final_overall_avg_df)

# Combine all result DataFrames into one, with empty lines in between
combined_df = pd.concat(result_dfs, ignore_index=True)

# Creating the output file name based on the input file names
output_file = "combined_average_percentage_by_case.csv"
combined_df.to_csv(output_file, index=False)
print(f"CSV file '{output_file}' created successfully.")
