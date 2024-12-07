import pandas as pd

# List of CSV file paths
csv_files = ["datasets/sickr-sts.tsv"]

# Define the prefix to filter columns
prefix = 'LLama3 ICL_4 (Sem Equiv)'

# Defining the bins and labels
bins = [0.00, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 4.99, 5.01]
labels = [
    "0-0.5",
    "0.5-1.0",
    "1.0-1.5",
    "1.5-2.0",
    "2.0-2.5",
    "2.5-3",
    "3.0-3.5",
    "3.5-4",
    "4-4.5",
    "4.5-4.99",
    "5.0"
]

# Process each CSV file
for file in csv_files:
    df = pd.read_csv(file, sep="\t")
    
    # Filter columns that start with the specified prefix
    columns_to_average = [col for col in df.columns if col.startswith(prefix)]
    
    if not columns_to_average:
        print(f"No columns found with prefix '{prefix}' in file '{file}'.")
        continue
    
    # Calculate the total count of 1s across the specified columns
    df['total_1s'] = df[columns_to_average].sum(axis=1)
    
    # Calculate the average number of 1s across all specified columns
    df['average_1s'] = df['total_1s'] / len(columns_to_average)

    # Creating the binned score column based on the original score
    df["score_bin"] = pd.cut(
        df["score"],
        bins=bins,
        labels=labels,
        right=False,
    )

    # Calculate the percentage of 1s for each row
    df['percentage_1s'] = df[columns_to_average].mean(axis=1) * 100

    # Group by the binned score and calculate required metrics
    bin_counts = df.groupby("score_bin").agg(
        count=('percentage_1s', 'count'),
        sum=('percentage_1s', 'sum'),
        std_dev=('percentage_1s', 'std'),
        avg_1s=('average_1s', 'mean')  # Calculate the average raw count of 1s
    )
    
    # Calculate the average number of 1s across all bins multiplied by the sample size
    bin_counts['avg_1s_weighted'] = bin_counts['avg_1s'] * bin_counts['count']

    bin_counts["average_percentage"] = (bin_counts["sum"] / bin_counts["count"])

    # Rounding the values to 3 decimal places
    bin_counts = bin_counts.round(3)

    # Creating the result DataFrame
    result_df = bin_counts.reset_index()

    # Delete the unnecessary columns
    result_df = result_df.drop(columns=["avg_1s", "sum"])

    # Rename the columns
    print(result_df.head())
    result_df.columns = [
        "Score Range",
        "Sample Size",
        "Standard Deviation of Percentages",
        "Weighted Average of 1s",
        "Average Percentage of 1s"
    ]

    # Reorder the columns
    result_df = result_df[[
        "Score Range",
        "Sample Size",
        "Weighted Average of 1s",
        "Average Percentage of 1s",
        "Standard Deviation of Percentages"
    ]]

    # Creating the output file name based on the input file name and prefix
    output_file = f"{file.split('.')[0]}_{prefix.replace(' ', '_')}_average_counts_percentage_bins.csv"
    result_df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' created successfully.")
