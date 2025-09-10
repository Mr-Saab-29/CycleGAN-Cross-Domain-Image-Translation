# CycleGAN-Cross-Domain-Image-Translation
**Notebook**: `Cross_Domain_Image_Translation_.ipynb`  
**Task**: Unpaired image-to-image translation on **apple2orange** using **CycleGAN** implemented from scratch in **PyTorch**.


---

## Overview
This notebook trains two generators and two discriminators to translate images **A→B** (apples→oranges) and **B→A** (oranges→apples) **without paired data**. It uses:
- **Generators**: ResNet-based (`n_blocks=9`) with reflection padding, instance norm, and Tanh output.
- **Discriminators**: **PatchGAN** (classifies overlapping image patches).
- **Losses**: Least-Squares GAN (**LSGAN**) with **label smoothing**, **cycle-consistency** L1 (λ=10), and **identity** loss.
- **Schedulers**: Linear LR decay beginning halfway through training.

---

##  Architecture (as in the notebook)
### Generators
```
GeneratorResNet(input_channel=3, output_channel=3, filters=64, n_blocks=9)
ReflectionPad2d(3) → Conv7x7 → IN → ReLU
↓ Downsample x2 (stride-2 convs)
→ 9 × Residual Blocks (Conv3x3 + IN + ReLU + skip)
↑ Upsample x2 (nearest + Conv/IN/ReLU or transposed conv)
ReflectionPad2d(3) → Conv7x7 → Tanh
```

### Discriminators (PatchGAN)
```
Discriminator(input_channel=3, filters=64)
Stack of Conv4x4 (stride 2) + IN + LeakyReLU
ZeroPad2d → final Conv to 1-channel map
```

### Losses
- **Adversarial** (LSGAN, MSE): use **soft labels**, e.g. real ∈ **[0.7,1.2]**, fake ∈ **[0.0,0.3]**.
- **Cycle-consistency**: `L_cycle = λ * (||x - G_YX(G_XY(x))||1 + ||y - G_XY(G_YX(y))||1)`, with **λ = 10**.
- **Identity**: `0.5 * λ * (||x - G_YX(x)||1 + ||y - G_XY(y)||1)`.

---

## Data
Dataset: **apple2orange** (from the official CycleGAN sets).  
The notebook **includes a cell** to download it via the Berkeley script and unpacks to:
```
datasets/apple2orange/
├─ trainA/   # domain A (apples)
├─ trainB/   # domain B (oranges)
├─ testA/
└─ testB/
```

**Transforms**  
- Train: `Resize(128) → RandomHorizontalFlip(0.5) → ToTensor() → Normalize((0.5,)*3, (0.5,)*3)`  
- Test:  `Resize(128) → ToTensor() → Normalize((0.5,)*3, (0.5,)*3)`

---

## Requirements
Python **3.10+** recommended.

```txt
torch
torchvision
numpy
pillow
matplotlib
tqdm
```
Install (CPU example):
```bash
pip install torch torchvision numpy pillow matplotlib tqdm
```
> For GPU acceleration, install CUDA-specific torch/torchvision wheels from pytorch.org matching your driver/CUDA version.

---

## How to Run
### Option A — Google Colab (recommended)
1. Open `Cross_Domain_Image_Translation_.ipynb` in Colab.
2. Run cells top-to-bottom. The dataset **download cell** will populate `datasets/apple2orange/`.
3. Ensure a GPU runtime (`Runtime → Change runtime type → T4/A100` if available).
4. Training artifacts (checkpoints, samples) will be saved to the paths below.

### Option B — Local (Jupyter / VS Code)
1. Create a virtual environment and install requirements.
2. Run the dataset download cell **or** manually place the folders under `datasets/apple2orange/`.
3. Launch Jupyter and run the notebook cells sequentially.

---

##  Training Configuration (from the notebook)
- `RESIZE_SHAPE = 128`
- `BATCH_SIZE = 10`
- `EPOCHs = 30`  *(note the variable name in the notebook)*
- `LR = 2e-4`, `BETAS = (0.5, 0.999)`
- **LR Scheduler**: `LambdaLR` with **linear decay** starting at `EPOCHs // 2` (after 15 epochs)
- **Training steps per epoch**: `ceil(min(len(trainA), len(trainB)) / BATCH_SIZE)`

### Checkpointing & Outputs
- **Checkpoints**: `checkpoints/pytorch/apple2orange/training-checkpoint`  
- **Sample images / logs**: under `outputs/` (and optionally `logs/pytorch/`)  
- **Save cadence**: every `SAVE_EVERY_N_EPOCH = 5` epochs (as configured in the notebook)

### Monitoring
- Prints averaged losses per epoch: `G_XtoY`, `G_YtoX`, `Dx`, `Dy`.
- Periodic visualization grids for: `A→B`, `B→A`, and cycle reconstructions.

---

##  Inference / Sampling (after or during training)
The notebook includes cells that:
- Load the latest checkpoint (if present)
- Generate samples for `A→B` and `B→A`
- Save visualization grids into `outputs/`

> If you stop and restart: run the **checkpoint load** cell before sampling to resume from the last saved state.

---

##  Tips & Troubleshooting
- **Mode collapse / Weird colors** early on is common—let training pass ~10–15 epochs.
- **CUDA OOM**: lower `BATCH_SIZE` (e.g., 4–8) or reduce `RESIZE_SHAPE` (e.g., 112/96).
- **Artifacts / checkerboards**: if present, consider switching upsampling mode or adding blur to upsampled features.
- **Slow training** on CPU: strongly recommend a GPU runtime (Colab T4/A100).

---

##  References
- CycleGAN: *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (Zhu et al., 2017)
- PatchGAN idea from *Image-to-Image Translation with Conditional Adversarial Networks* (Isola et al., 2017)

---

## License
Academic/educational use aligned with your coursework or personal experimentation.
