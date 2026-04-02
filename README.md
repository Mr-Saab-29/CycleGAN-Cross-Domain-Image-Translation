# CycleGAN Cross-Domain Image Translation

This repository has been converted from a notebook into a runnable end-to-end PyTorch project for unpaired image-to-image translation with CycleGAN. The original notebook is still present for reference, but the modular code under `src/` is now the primary execution path.

## Project Structure

```text
.
├── Cross_Domain_Image_Translation_.ipynb
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── download_dataset.py
│   ├── generate_samples.py
│   └── train.py
└── src/
    └── cyclegan/
        ├── checkpoints.py
        ├── config.py
        ├── data.py
        ├── inference.py
        ├── losses.py
        ├── models.py
        ├── trainer.py
        └── utils.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## End-to-End Usage

1. Download the dataset.

```bash
python scripts/download_dataset.py --dataset-name apple2orange
```

2. Train the model.

```bash
python scripts/train.py --dataset-name apple2orange --epochs 100 --batch-size 1 --image-size 256
```

3. Generate qualitative samples from the latest checkpoint.

```bash
python scripts/generate_samples.py --dataset-name apple2orange --sample-index 0
```

## Outputs

- Checkpoints: `checkpoints/pytorch/<experiment-name>/`
- Training metrics: `logs/pytorch/<experiment-name>/metrics.csv`
- Training previews: `outputs/<experiment-name>/samples/`
- Inference previews: `outputs/<experiment-name>/inference/`

## Notes

- Training resumes automatically from the latest checkpoint unless `--no-resume` is passed.
- Default training is now closer to standard CycleGAN practice: full dataset, `256x256`, `batch_size=1`, `100` epochs.
- The trainer uses a replay buffer for fake images to stabilize discriminator updates.
- Pass `--subset-size 400` if you want the older faster notebook-style subset behavior.
