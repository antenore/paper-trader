#!/usr/bin/env bash
set -euo pipefail

echo "=== Paper Trader — Setup ==="

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it to add your Anthropic API key"
fi

# Create data directory
mkdir -p data

# Install dependencies
echo "Installing dependencies..."
uv sync

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and set PT_ANTHROPIC_API_KEY"
echo "  2. Run: uv run paper-trader"
echo "  3. Open: http://0.0.0.0:8420"
echo ""
