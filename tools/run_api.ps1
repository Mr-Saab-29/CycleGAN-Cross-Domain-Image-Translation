param(
    [string]$Dataset = "apple2orange",
    [string]$Experiment = "apple2orange",
    [string]$ImageSize = "256",
    [string]$Device = "cpu",
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000
)

$env:MODEL_DATASET_NAME = $Dataset
$env:MODEL_EXPERIMENT_NAME = $Experiment
$env:MODEL_CHECKPOINT_ROOT = "checkpoints/pytorch"
$env:MODEL_OUTPUT_ROOT = "outputs"
$env:MODEL_IMAGE_SIZE = $ImageSize
$env:MODEL_DEVICE = $Device

python -m uvicorn app.main:app --host $Host --port $Port
