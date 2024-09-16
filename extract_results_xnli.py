import pandas as pd

# List of CSV file paths
csv_files = ["datasets/fb-xnli-hyp-pre.tsv", "datasets/fb-xnli-pre-hyp.tsv"]

# Dictionary of prefixes to process for each file
prefixes_to_process = {
    "datasets/fb-xnli-hyp-pre.tsv": "LLama3 ICL_4 (Sem Equiv)",
    "datasets/fb-xnli-pre-hyp.tsv": "LLama3 ICL_4 (Sem Equiv)"
}

# Initialize dictionaries to store combined results and percentage calculations
combined_results = {"contradiction": {}, "neutral": {}}
percentages_by_lang_case = {"contradiction": {}, "neutral": {}}
overall_sums = {"contradiction": {"count": 0, "sum": 0}, "neutral": {"count": 0, "sum": 0}}
division_by = 0

# Process each CSV file
for file in csv_files:
    prefix = prefixes_to_process[file]
    # Load the file once to avoid multiple reads
    df = pd.read_csv(file, sep="\t")
    # Identify columns with the specified prefix
    columns_to_process = [col for col in df.columns if col.startswith(prefix)]
    division_by += len(columns_to_process)  # Track the number of columns for division

    # Iterate over the columns to process
    for column in columns_to_process:
        for score_value, case_name in [(-1, "contradiction"), (0, "neutral")]:
            # Filter data based on the score value and group by language
            filtered_df = df[df["score"] == score_value]
            lang_counts = filtered_df.groupby("lang")[column].agg(["count", "sum"])
            lang_counts["percentage"] = (lang_counts["sum"] / lang_counts["count"]) * 100

            # Process results for each language
            for lang in lang_counts.index:
                count = lang_counts.loc[lang, "count"]
                sum_vals = lang_counts.loc[lang, "sum"]
                percentage = lang_counts.loc[lang, "percentage"]

                # Initialize combined results if needed
                if lang not in combined_results[case_name]:
                    combined_results[case_name][lang] = {"count": 0, "sum": 0}

                # Update combined counts and sums
                combined_results[case_name][lang]["count"] += count
                combined_results[case_name][lang]["sum"] += sum_vals

                # Store percentages for calculating standard deviation
                if lang not in percentages_by_lang_case[case_name]:
                    percentages_by_lang_case[case_name][lang] = []
                percentages_by_lang_case[case_name][lang].append(percentage)

                # Aggregate overall sums for the final calculation
                overall_sums[case_name]["count"] += count
                overall_sums[case_name]["sum"] += sum_vals

# Prepare output rows and calculate overall statistics
output_rows = []
for case_name, lang_data in combined_results.items():
    for lang, data in lang_data.items():
        percentage = (data["sum"] / data["count"]) * 100
        lang_std_dev = pd.Series(percentages_by_lang_case[case_name][lang]).std()
        output_rows.append([lang, case_name, data["count"], data["sum"], round(percentage, 3), round(lang_std_dev, 3)])

# Calculate overall averages and standard deviations
for case_name, totals in overall_sums.items():
    overall_percentage = (totals["sum"] / totals["count"]) * 100
    overall_std_dev = pd.Series(
        [val for lang_percentages in percentages_by_lang_case[case_name].values() for val in lang_percentages]
    ).std()
    output_rows.append(["Overall Average", case_name, totals["count"], totals["sum"], round(overall_percentage, 3), round(overall_std_dev, 3)])

# Convert to DataFrame and assign column names
final_df = pd.DataFrame(output_rows, columns=["Language", "Case", "Sample Count", "Count of 1s", "Percentage of 1s", "Standard Deviation of 1s"])

# Adjust sample counts and counts of 1s by dividing by the number of columns processed
final_df["Sample Count"] = final_df["Sample Count"] / division_by
final_df["Count of 1s"] = final_df["Count of 1s"] / division_by

# Save the results to a TSV file
final_df.to_csv("xnli_results_averaged_multiencoder.tsv", sep="\t", index=False)

# Display final DataFrame to check results
print(final_df)