#!/bin/bash
# Start both backend and frontend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
