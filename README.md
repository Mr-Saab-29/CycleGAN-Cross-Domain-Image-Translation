# CycleGAN Cross-Domain Image Translation

This repository now supports both plain training and a minimal end-to-end MLOps workflow for CycleGAN image translation. The core model remains the same, but the project now includes config-driven experiment runs, optional MLflow tracking, reproducible evaluation, single-image inference, and a deployable FastAPI service.

## Project Structure

```text
.
├── app/
│   └── main.py
├── configs/
│   └── apple2orange.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── generate_samples.py
│   ├── infer_image.py
│   ├── run_experiment.py
│   └── train.py
├── src/
│   └── cyclegan/
│       ├── checkpoints.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── losses.py
│       ├── models.py
│       ├── tracking.py
│       ├── trainer.py
│       └── utils.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

```bash
python scripts/download_dataset.py --dataset-name apple2orange
```

## Standard Training

```bash
python scripts/train.py --dataset-name apple2orange
```

This saves:

- Checkpoints in `checkpoints/pytorch/<run-name>/`
- Epoch metrics in `logs/pytorch/<run-name>/metrics.csv`
- Training previews in `outputs/<run-name>/samples/`
- Evaluation artifacts in `outputs/<run-name>/evaluation/`

## Config-Driven MLOps Run

Use the YAML config when you want a reproducible tracked run:

```bash
python scripts/run_experiment.py --config configs/apple2orange.yaml
```

What this adds:

- Config-defined experiment parameters
- MLflow tracking if `mlflow` is installed and `tracking_enabled: true`
- Automatic evaluation report after training
- Evaluation preview image and JSON report saved with the run

Default MLflow tracking output is stored under `mlruns/`.

## Evaluation

Run evaluation separately against a trained checkpoint:

```bash
python scripts/evaluate.py --dataset-name apple2orange
```

This writes:

- `outputs/<run-name>/evaluation/report.json`
- `outputs/<run-name>/evaluation/evaluation_preview.png`

## Inference

### Dataset Test-Split Preview

```bash
python scripts/generate_samples.py --dataset-name apple2orange --sample-index 0
```

### Single Image Inference

```bash
python scripts/infer_image.py --dataset-name apple2orange --direction x2y --input input.jpg --output translated.png
```

Use `--direction y2x` for the reverse translation.

## API Serving

Start the FastAPI service locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set model settings with environment variables if needed:

```bash
set MODEL_DATASET_NAME=apple2orange
set MODEL_EXPERIMENT_NAME=apple2orange
set MODEL_CHECKPOINT_ROOT=checkpoints/pytorch
set MODEL_IMAGE_SIZE=256
set MODEL_DEVICE=cuda
```

API endpoints:

- `GET /health`
- `POST /predict?direction=x2y`
- `POST /predict?direction=y2x`

Example request:

```bash
curl -X POST "http://localhost:8000/predict?direction=x2y" -F "file=@input.jpg" --output translated.png
```

## Docker

Build the inference container:

```bash
docker build -t cyclegan-api .
```

Run it:

```bash
docker run -p 8000:8000 ^
  -e MODEL_DATASET_NAME=apple2orange ^
  -e MODEL_EXPERIMENT_NAME=apple2orange ^
  -e MODEL_DEVICE=cpu ^
  -v %cd%\\checkpoints:/app/checkpoints ^
  cyclegan-api
```

Or use Docker Compose:

```bash
docker compose up --build
```

The compose file is [docker-compose.yml](c:/Users/sabar/Desktop/Documents/Personal/Image%20Translation/CycleGAN-Cross-Domain-Image-Translation/docker-compose.yml) and mounts:

- `./checkpoints` to `/app/checkpoints`
- `./outputs` to `/app/outputs`

If you trained under a different run name, update `MODEL_EXPERIMENT_NAME` in the compose file before starting the container.

## Notes

- The project supports other unpaired datasets with the same folder layout: `trainA`, `trainB`, `testA`, `testB`.
- For long local GPU training runs, `batch_size=1` and `image_size=256` are the intended defaults.
- MLflow tracking is optional. If it is not installed, training still runs, but experiment tracking is skipped.
