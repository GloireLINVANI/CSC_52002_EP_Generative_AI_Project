#!/bin/bash

# Output CSV file
output_file="fid_scores.csv"

# Write the header to the CSV file
echo "dataset,mask,comparison,FID" > $output_file

# Loop through each dataset and mask
for dataset in samples/*; do
    for mask in "$dataset"/*; do
        # Extract dataset and mask names
        dataset_name=$(basename "$dataset")
        mask_name=$(basename "$mask")

        # Define the subfolders
        subfolders=("base" "repaint" "rpip")

        # Run the FID calculations and append the results to the CSV file
        for subfolder in "${subfolders[@]}"; do
            fid_output=$(python -m pytorch_fid "$mask/original" "$mask/$subfolder")
            fid_score=$(echo "$fid_output" | grep -oP 'FID:  \K[0-9.]+')
            echo "$dataset_name,$mask_name,$subfolder,$fid_score" >> $output_file
        done
    done
done