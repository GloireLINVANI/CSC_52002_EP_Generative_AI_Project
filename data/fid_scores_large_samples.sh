#!/bin/bash

# Output CSV file
output_file="fid_scores_large.csv"

# Write the header to the CSV file
echo "dataset,comparison,FID" > $output_file

# Loop through each dataset and mask
for dataset in large_samples/*; do
    # Extract dataset and mask names
    dataset_name=$(basename "$dataset")

    # Define the subfolders
    subfolders=("base" "repaint" "rpip")

    # Run the FID calculations and append the results to the CSV file
    for subfolder in "${subfolders[@]}"; do
        fid_output=$(python -m pytorch_fid "$dataset/original" "$dataset/$subfolder")
        fid_score=$(echo "$fid_output" | grep -oP 'FID:  \K[0-9.]+')
        echo "$dataset_name,$subfolder,$fid_score" >> $output_file
    done
done