#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8501}"

python -m streamlit run app/streamlit_app.py --server.port "$PORT"
