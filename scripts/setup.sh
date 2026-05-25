#!/bin/bash

# Create folders
folders=("data" "logs" "outputs" "results")

for folder in "${folders[@]}"; do
    if [ ! -d "$folder" ]; then
        mkdir "$folder"
        echo "Created folder: $folder"
    else
        echo "Folder already exists: $folder"
    fi
done
echo "Folders created successfully"


env_file=".env"

cat > "$env_file" << EOL
PROJECT_ROOT="."
RES_DIR="." # set to the desired location
DATA_DIR="\${RES_DIR}/data"
LOG_DIR="\${RES_DIR}/logs"
OUTPUT_DIR="\${RES_DIR}/outputs"
EOL

echo ".env file created successfully at $PWD/$env_file"