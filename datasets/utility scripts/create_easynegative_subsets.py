import pandas as pd
import numpy as np

# Sample DataFrame creation for demonstration (remove this if you already have a DataFrame)
# Replace this with your actual data loading method

# Extract the middle number from the swap_id
df_swap = pd.read_csv("input_backtransl_wiki_with_swap_id.tsv", sep="\t")
# Step 2: Clean the swap_id column
# Remove any leading/trailing whitespace and drop rows where swap_id is NaN or does not have the expected format
df_swap['swap_id'] = df_swap['swap_id'].astype(str).str.strip()

# Step 3: Extract the middle number safely
def extract_middle_number(swap_id):
    try:
        parts = swap_id.split('_')
        if len(parts) == 3:  # Ensure there are enough parts after splitting
            return parts[-2]
        else:
            return -1
    except Exception as e:
        print(f"Error processing swap_id '{swap_id}': {e}")
        return None

df_swap['middle_number'] = df_swap['swap_id'].apply(extract_middle_number)

# Remove any rows where middle_number could not be extracted
df_swap = df_swap.dropna(subset=['middle_number'])
# Step 4: Create the dictionary of id to middle number
id_to_middle_number = dict(zip(df_swap['id'], df_swap['middle_number']))
df = pd.read_csv('relabeled_train_paws_x_en.tsv', sep="\t")
# Step 2: Shuffle the dataset with a pre-set random seed for reproducibility
seed = 42
df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Step 3: Filter data where label is 0 and label is 1 separately after shuffling
df_label_0 = df_shuffled[df_shuffled['label'] == 0].reset_index(drop=True)
df_label_1 = df_shuffled[df_shuffled['label'] == 1].reset_index(drop=True)

# Step 4: Determine subset sizes (25%, 50%, 75%, 100%) for label 0 samples
total_size_label_0 = len(df_label_0)
subset_sizes = [int(total_size_label_0 * 0.25), int(total_size_label_0 * 0.50), int(total_size_label_0 * 0.75), total_size_label_0]

def trace_backtranslate_to_swap(mapping):
    return id_to_middle_number.get(mapping[:-2], -1)

# Step 5: Function to create subsets, swap sentence2 with a random sentence1, include label 1 samples, retain remaining original label 0 samples, and add 'is_augmented' column
def create_swapped_subsets_with_label_1(df_label_0, df_label_1, subset_sizes, seed):
    subsets = []  # To store the different subsets (25%, 50%, 75%, 100%)
    
    # Random indices for random selection without replacement
    np.random.seed(seed)  # Set random seed for reproducibility
    all_indices = np.random.permutation(len(df_label_0))

    # Iteratively create each subset
    for i, size in enumerate(subset_sizes):
        # Select indices for the current subset size for augmented samples
        current_indices = all_indices[:size]
        
        # Create the current subset with label 0 samples for augmentation
        current_subset_label_0 = df_label_0.iloc[current_indices].copy()
        current_subset_label_0['is_negative_and_augmented'] = 1  # Mark these as augmented

        # Swap sentence2 with another randomly selected sentence1 from either df_label_0 or df_label_1
        for idx in current_subset_label_0.index:
            mapping_check = current_subset_label_0.at[idx, 'mapping1'].split("_")[0]
            if mapping_check == "backtransl":
                original_mapping1 = trace_backtranslate_to_swap(current_subset_label_0.at[idx, 'mapping1'])
            else:
                original_mapping1 = current_subset_label_0.at[idx, 'mapping1'].split("_")[-2]  # Original mapping1 ID
            
            # Ensure the new sentence1's mapping1 is not the same as the original mapping2
            while True:
                # Randomly select another sentence1 from either df_label_0 or df_label_1
                random_row = pd.concat([df_label_0, df_label_1]).sample(n=1, random_state=seed).iloc[0]
                random_sentence2 = random_row['sentence2']

                mapping_check = random_row['mapping2'].split("_")[0]
                if mapping_check == "backtransl":
                    random_mapping2  = trace_backtranslate_to_swap(random_row['mapping2'])
                else:
                    random_mapping2 = random_row['mapping2'].split("_")[-2]  # Original mapping1 ID
                
                # Check if the mapping1 of the selected sentence1 is different from the original mapping1
                if random_mapping2 != original_mapping1:
                    break  # Exit the loop if a different mapping2 is found
                else:
                    print('IDENTICAL MAPPING')
                    seed = seed + 1
            
            # Replace sentence2 with the randomly selected sentence1
            current_subset_label_0.at[idx, 'sentence2'] = random_sentence2
        
        # Include the remaining label 0 samples with their original sentence2 and mark them as not augmented
        remaining_label_0_indices = all_indices[size:]  # Remaining indices for label 0
        remaining_label_0_subset = df_label_0.iloc[remaining_label_0_indices].copy()
        remaining_label_0_subset['is_negative_and_augmented'] = 0
        
        # Combine modified subset with remaining label 0 samples
        current_combined_label_0_subset = pd.concat([current_subset_label_0, remaining_label_0_subset], ignore_index=True)
        
        # Mark all label 1 samples as not augmented
        df_label_1_copy = df_label_1.copy()
        df_label_1_copy['is_negative_and_augmented'] = 0
        
        # Combine the modified subset with all label 1 samples
        combined_subset = pd.concat([current_combined_label_0_subset, df_label_1_copy], ignore_index=True)
        
        # Re-sort to original order using 'id'
        combined_subset_sorted = combined_subset.sort_values('id').reset_index(drop=True)
        
        # Add the combined subset to the list of subsets
        subsets.append(combined_subset_sorted)
        
    return subsets

# Step 6: Apply the function to create the subsets
subsets = create_swapped_subsets_with_label_1(df_label_0, df_label_1, subset_sizes, seed)

# Step 7: Save the results to CSV files
for i, subset in enumerate(subsets):
    subset_size_percentage = int((i + 1) * 25)
    filename = f"subset_{subset_size_percentage}_percent.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved {filename}")