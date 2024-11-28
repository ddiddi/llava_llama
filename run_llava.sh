#!/bin/bash

# Name: run_llama.sh
# Purpose: Executes a Python script for LLama multi-modal model

# Ensure script is run with bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script must be run using bash."
  exit 1
fi

# Define Python environment or command
PYTHON_CMD=python3

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
  echo "Python is not installed. Please install Python 3."
  exit 1
fi

# Check if required dependencies are installed
REQUIRED_LIBS=("llama_cpp" "base64")

echo "Checking dependencies..."
for lib in "${REQUIRED_LIBS[@]}"; do
  $PYTHON_CMD -c "import $lib" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "Dependency $lib is not installed. Installing..."
    $PYTHON_CMD -m pip install $lib
    if [ $? -ne 0 ]; then
      echo "Failed to install $lib. Please check your Python environment."
      exit 1
    fi
  fi
done
echo "All dependencies are satisfied."

# Run the Python script
echo "Running LLama multi-modal model..."
$PYTHON_CMD run_llava.py

# Exit with success
exit 0
