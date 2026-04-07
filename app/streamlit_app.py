from __future__ import annotations

import io
from pathlib import Path
import sys

import streamlit as st
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cyclegan.config import CycleGANConfig
from cyclegan.inference import build_inference_transform, load_generators_for_inference, tensor_to_pil, translate


st.set_page_config(page_title="CycleGAN Translator", page_icon="🍎", layout="wide")


@st.cache_resource(show_spinner=False)
def load_models(dataset_name: str, experiment_name: str, image_size: int, device: str):
    config = CycleGANConfig(
        dataset_name=dataset_name,
        checkpoint_root=Path("checkpoints") / "pytorch",
        output_root=Path("outputs"),
        image_size=image_size,
        experiment_name=experiment_name,
        device=device,
        subset_size=None,
        tracking_enabled=False,
    )
    return config, load_generators_for_inference(config)


def main() -> None:
    st.title("CycleGAN Domain Translator")
    st.caption("Upload an apple or orange image and translate it to the other domain.")

    with st.sidebar:
        st.header("Model Settings")
        dataset_name = st.text_input("Dataset name", value="apple2orange")
        experiment_name = st.text_input("Experiment name", value="apple2orange")
        image_size = st.number_input("Image size", min_value=64, max_value=512, value=256, step=32)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = st.selectbox("Device", options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0 if default_device == "cpu" else 1)
        direction = st.radio("Translation direction", options=("x2y", "y2x"), horizontal=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        config, (generator_x_to_y, generator_y_to_x) = load_models(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            image_size=int(image_size),
            device=device,
        )
    except FileNotFoundError:
        st.error(f"Checkpoint not found at {Path('checkpoints') / 'pytorch' / experiment_name / 'training-checkpoint.pt'}")
        return
    except Exception as exc:
        st.error(f"Failed to load the model: {exc}")
        return

    image = Image.open(uploaded_file).convert("RGB")
    tensor = build_inference_transform(config.image_size)(image).unsqueeze(0)
    model = generator_x_to_y if direction == "x2y" else generator_y_to_x
    translated = translate(model, tensor, config.device)
    translated_image = tensor_to_pil(translated)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Translated")
        st.image(translated_image, use_container_width=True)

    buffer = io.BytesIO()
    translated_image.save(buffer, format="PNG")
    st.download_button(
        label="Download translated image",
        data=buffer.getvalue(),
        file_name=f"{direction}_translated.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
