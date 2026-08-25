#!/bin/bash
# Setup script for Compliance-Aware RAG
# Usage: ./setup.sh

set -e

echo "=== Compliance-Aware RAG Setup ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required."
    exit 1
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set."
    echo "You'll need it to build the graph and FAISS index."
    echo "Set it with: export OPENAI_API_KEY=sk-..."
    echo ""
fi

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "2. Installing dependencies..."
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Check if graph exists
GRAPH_PATH="../data/graph_frozen.pkl"
if [ ! -f "$GRAPH_PATH" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo ""
        echo "NOTICE: Knowledge graph not found and no API key set."
        echo "Run this after setting OPENAI_API_KEY:"
        echo "  cd backend && python build_graph.py"
    else
        echo ""
        echo "3. Building knowledge graph (this takes ~5 min and ~365 API calls)..."
        cd backend && python build_graph.py && cd ..
    fi
else
    echo "3. Knowledge graph found."
fi

# Check if FAISS index exists
FAISS_PATH="../data/faiss_index/index.faiss"
if [ ! -f "$FAISS_PATH" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo ""
        echo "NOTICE: FAISS index not found and no API key set."
        echo "Run this after setting OPENAI_API_KEY:"
        echo "  cd backend && python build_index.py"
    else
        echo ""
        echo "4. Building FAISS index..."
        cd backend && python build_index.py && cd ..
    fi
else
    echo "4. FAISS index found."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the app:"
echo "  # Terminal 1 - Backend"
echo "  cd app && source .venv/bin/activate"
echo "  uvicorn backend.main:app --reload --port 8000"
echo ""
echo "  # Terminal 2 - Frontend"
echo "  cd app && source .venv/bin/activate"
echo "  streamlit run frontend/streamlit_app.py --server.port 8501"
echo ""
echo "Then open http://localhost:8501 in your browser."
