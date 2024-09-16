#!/bin/bash

# Path to your Python script
SCRIPT="full_experimentation_pipeline.py"

# Directory containing your config files
CONFIG_DIR="configs/multi_encoders_orig" # source directory to run all the json files for

# Loop through all JSON files in the config directory
for CONFIG in $CONFIG_DIR/*.json
do
    # Run the Python script with the current config file
    echo "Running experiment with $CONFIG"
    python3 $SCRIPT $CONFIG

    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Experiment with $CONFIG failed"
        exit 1
    fi
done

echo "All experiments completed successfully."
