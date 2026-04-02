FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts

ENV PYTHONPATH=/app/src
ENV MODEL_DATASET_NAME=apple2orange
ENV MODEL_EXPERIMENT_NAME=apple2orange
ENV MODEL_CHECKPOINT_ROOT=checkpoints/pytorch
ENV MODEL_OUTPUT_ROOT=outputs
ENV MODEL_IMAGE_SIZE=256
ENV MODEL_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
