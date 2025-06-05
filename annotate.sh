#!/bin/bash

# Ensure virtual environment is activated
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "⚠️  Virtual environment is not active. Run 'source .venv/bin/activate' first."
    exit 1
fi

# Set PYTHONPATH to the project root
export PYTHONPATH=$PWD

# Run the CLI tool with arguments
python -m cli.main \
    --input "$1" \
    --output "$2" \
    --toc-json "$3" \
    --language "${4:-en}"
