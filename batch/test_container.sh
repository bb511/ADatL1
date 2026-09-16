#!/bin/bash

set -euo pipefail

echo "========================================"
echo "HTCondor SIF test"
echo "========================================"

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "PWD:      $(pwd)"
echo "User:     $(whoami)"

echo
echo "Python:"
command -v python
python --version

cd /eos/user/l/lbehrens/adatl1/ADatL1
echo "Switched to /eos/user/l/lbehrens/adatl1/ADatL1"

echo
echo "Checking test.py:"
ls -lh test.py

echo
echo "Running test.py ..."
python test.py

echo
echo "========================================"
echo "TEST FINISHED SUCCESSFULLY"
echo "========================================"
