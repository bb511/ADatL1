#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

REQUIREMENTS_FILE="requirements.txt"

# Create temporary processed file without line breaks
processed_requirements=$(mktemp)
sed ':a; /\\$/ { N; s/\\\n//; ba }; s/ *; */;/g' "$REQUIREMENTS_FILE" > "$processed_requirements"

echo -e "\nReading requirements from ${REQUIREMENTS_FILE} (processed version)...\n"
failed_packages=()

while IFS= read -r package || [ -n "$package" ]; do
    # Skip empty lines and comments
    if [[ -z "$package" || "$package" == \#* ]]; then
        continue
    fi

    echo -e "Installing package: ${package}..."

    # Attempt to install the package
    python -m pip install --require-hashes "$package"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed: ${package}${NC}"
    else
        echo -e "${RED}Failed to install: ${package}${NC}"
        failed_packages+=("$package")
    fi
    echo
done < "$processed_requirements"

# Cleanup temporary file
rm "$processed_requirements"

# Report failures
if [ ${#failed_packages[@]} -ne 0 ]; then
    echo -e "\n${RED}Failed packages:${NC}"
    printf ' - %s\n' "${failed_packages[@]}"
    exit 1
fi

echo -e "\n${GREEN}All packages installed successfully!${NC}"
exit 0