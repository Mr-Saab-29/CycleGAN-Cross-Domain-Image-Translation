param(
    [int]$Port = 8501
)

python -m streamlit run app/streamlit_app.py --server.port $Port
