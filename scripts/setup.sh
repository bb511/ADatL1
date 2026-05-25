#!/bin/bash

# Create folders
folders=("data" "logs" "outputs" "results" "checkpoints" "data/raw")

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
PROJECT_ROOT="$PWD"
RES_DIR="$PWD"
DATA_DIR="\${RES_DIR}/data"
LOG_DIR="\${RES_DIR}/logs"
OUTPUT_DIR="\${RES_DIR}/outputs"
CHECKPOINT_DIR="\${RES_DIR}/checkpoints"
RAW_DATA_DIR="\${DATA_DIR}/raw"
HYDRA_FULL_ERROR=1
WANDB_MODE=offline
EOL

echo ".env file created successfully at $PWD/$env_file"
