from __future__ import annotations

import os
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cyclegan.config import CycleGANConfig
from cyclegan.inference import image_bytes_to_tensor, load_generators_for_inference, tensor_to_pil, translate


def build_config_from_env() -> CycleGANConfig:
    return CycleGANConfig(
        dataset_name=os.getenv("MODEL_DATASET_NAME", "apple2orange"),
        checkpoint_root=Path(os.getenv("MODEL_CHECKPOINT_ROOT", "checkpoints/pytorch")),
        output_root=Path(os.getenv("MODEL_OUTPUT_ROOT", "outputs")),
        image_size=int(os.getenv("MODEL_IMAGE_SIZE", "256")),
        experiment_name=os.getenv("MODEL_EXPERIMENT_NAME") or None,
        device=os.getenv("MODEL_DEVICE", "cpu"),
        subset_size=None,
        tracking_enabled=False,
    )


app = FastAPI(title="CycleGAN Inference API", version="0.1.0")
config = build_config_from_env()
generator_x_to_y = None
generator_y_to_x = None


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "message": "CycleGAN inference API is running.",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict?direction=x2y or /predict?direction=y2x",
        }
    )


@app.on_event("startup")
def load_models() -> None:
    global generator_x_to_y, generator_y_to_x
    if not config.checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found at {config.checkpoint_path}")
    generator_x_to_y, generator_y_to_x = load_generators_for_inference(config)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "dataset_name": config.dataset_name,
            "experiment_name": config.run_name,
            "device": config.device,
            "checkpoint_path": str(config.checkpoint_path),
        }
    )


@app.get("/demo", response_class=Response)
def demo() -> Response:
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>CycleGAN Demo</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }
        form { display: grid; gap: 1rem; }
        button { width: fit-content; padding: 0.6rem 1rem; }
        .hint { color: #555; }
      </style>
    </head>
    <body>
      <h1>CycleGAN Demo</h1>
      <p class="hint">Upload an image and choose a translation direction.</p>
      <form action="/predict" method="post" enctype="multipart/form-data">
        <label>
          Direction:
          <select name="direction">
            <option value="x2y">x2y</option>
            <option value="y2x">y2x</option>
          </select>
        </label>
        <label>
          Image:
          <input type="file" name="file" accept="image/*" required>
        </label>
        <button type="submit">Translate</button>
      </form>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    direction: str = Query(..., pattern="^(x2y|y2x)$"),
) -> Response:
    if generator_x_to_y is None or generator_y_to_x is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model = generator_x_to_y if direction == "x2y" else generator_y_to_x
    tensor = image_bytes_to_tensor(image_bytes, config.image_size)
    translated = translate(model, tensor, config.device)
    image = tensor_to_pil(translated)

    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
