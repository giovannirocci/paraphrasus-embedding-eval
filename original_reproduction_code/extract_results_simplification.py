import pandas as pd

# List of CSV file paths for non-multilingual datasets
csv_files = ["datasets/onestop_parallel_all_pairs.tsv"]

# List of columns to process for each file (can be different for each file)
columns_to_process = {
    "datasets/facebook-anli.tsv": [
    "XLM-RoBERTa-EN-ALLTHREE-V3-1",
    "XLM-RoBERTa-EN-ALLTHREE-V3-2",
    "XLM-RoBERTa-EN-ALLTHREE-V3-3",
    "XLM-RoBERTa-EN-ALLTHREE-V3-4",
],
    "datasets/stanfordnlp-snli.tsv": [
    "XLM-RoBERTa-EN-ALLTHREE-V3-1",
    "XLM-RoBERTa-EN-ALLTHREE-V3-2",
    "XLM-RoBERTa-EN-ALLTHREE-V3-3",
    "XLM-RoBERTa-EN-ALLTHREE-V3-4",
],
}


# Process each CSV file
for file in csv_files:
    df = pd.read_csv(file, sep="\t")
    
    # List to store the result DataFrames for each column
    result_dfs = []
    
    # Process each specified column in the current file
    for column in columns_to_process[file]:
        # Iterate over the specified SourceLevel and TargetLevel combinations
        for source_level, target_level in level_combinations:
            # Filter the DataFrame based on the SourceLevel and TargetLevel combination
            filtered_df = df[(df["SourceLevel"] == source_level) & (df["TargetLevel"] == target_level)]

            # Calculate required metrics
            total_count = filtered_df[column].count()
            total_sum = filtered_df[column].sum()
            percentage = (total_sum / total_count) * 100 if total_count > 0 else 0

            # Rounding the percentage to 3 decimal places
            percentage = round(percentage, 3)

            # Creating the result DataFrame
            result_df = pd.DataFrame({
                "Metric": [f"{column} ({source_level} {target_level})"],
                "Sample Count": [total_count],
                "Count of 1s": [total_sum],
                "Percentage of 1s": [percentage]
            })

            # Append the result DataFrame to the list
            result_dfs.append(result_df)

            # Append an empty DataFrame for the empty line
            result_dfs.append(pd.DataFrame({"Metric": [""], "Sample Count": [""], "Count of 1s": [""], "Percentage of 1s": [""]}))

    # Combine all result DataFrames into one, with empty lines in between
    combined_df = pd.concat(result_dfs, ignore_index=True)

    # Creating the output file name based on the input file name
    output_file = f"{file.split('.')[0]}_combined_percentage.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' created successfully.")