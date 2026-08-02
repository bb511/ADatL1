#!/bin/bash
# One-time project bootstrap: create the working directories and write .env.
#
# configs/paths/default.yaml reads ${oc.env:PROJECT_ROOT} from .env, so nothing
# in the repository composes until this has run. Run it from anywhere; it always
# sets the repository root up, not the current directory.
#
#   bash scripts/setup.sh
#
# To keep the heavy directories on another filesystem, edit RES_DIR in .env
# afterwards and run scripts/symbolink.sh, which replaces them with symlinks.
set -euo pipefail

cd "$(dirname "$0")/.."

# checkpoints/ is written by the Lightning callbacks; the other three are the
# roots configs/paths/default.yaml hands to hydra.
for folder in data logs outputs checkpoints; do
    if [ -d "$folder" ]; then
        echo "Folder already exists: $folder"
    else
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done

# Never silently replace an existing .env: it may already point RES_DIR at a
# scratch filesystem, and losing that quietly means the next run fills the home
# directory with checkpoints.
env_file=".env"
if [ -e "$env_file" ]; then
    echo
    echo "$PWD/$env_file already exists - left untouched."
    echo "Delete it first if you want a fresh one."
    exit 0
fi

cat > "$env_file" << EOL
PROJECT_ROOT="."
RES_DIR="." # set to the desired location, then run scripts/symbolink.sh
DATA_DIR="\${RES_DIR}/data"
LOG_DIR="\${RES_DIR}/logs"
OUTPUT_DIR="\${RES_DIR}/outputs"
CHECKPOINT_DIR="\${RES_DIR}/checkpoints"
EOL

echo
echo ".env file created successfully at $PWD/$env_file"
