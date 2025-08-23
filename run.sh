#!/bin/bash
# Run script for Decision Matrix MCP server using uv

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Enable debug logging if DEBUG environment variable is set
if [ "$DEBUG" = "1" ]; then
    export LOG_LEVEL="DEBUG"
    echo "Debug logging enabled" >&2
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..." >&2
    uv venv
fi

# Install the package in development mode
echo "Installing dependencies..." >&2
uv pip install -e . >&2

# Run the server
echo "Starting Decision Matrix MCP server..." >&2
uv run python -m decision_matrix_mcp
