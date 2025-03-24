#!/bin/bash

# Check if tar file and file count are provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <tar_file> [files_per_class]"
    exit 1
fi

TAR_FILE=$1
FILES_PER_CLASS=${2:-5}  # Default to 5 if not provided

# Cache the list of files in the archive to avoid multiple tar calls
echo "Caching file list from $TAR_FILE..."
FILE_LIST=$(mktemp)
tar tf "$TAR_FILE" > "$FILE_LIST"
echo "File list cached."

# List unique class directories using wildcard matching
echo "Identifying class directories..."
CLASS_DIRS=$(grep '^[^/]*/[^/]*/[^/]*/' "$FILE_LIST" | awk -F'/' '{print $1"/"$2"/"$3}' | sort -u)
echo "Found $(echo "$CLASS_DIRS" | wc -l) class directories."

# Extract a few files per class
echo "Starting extraction..."
for class_dir in $CLASS_DIRS; do
    echo "Processing: $class_dir"
    grep "^$class_dir/" "$FILE_LIST" | shuf | head -n $FILES_PER_CLASS | xargs -d '\n' tar xf "$TAR_FILE"
    echo "Extracted $FILES_PER_CLASS files from $class_dir."
done

# Cleanup
echo "Cleaning up temporary files..."
test -f "$FILE_LIST" && rm "$FILE_LIST"
echo "Extraction complete."
