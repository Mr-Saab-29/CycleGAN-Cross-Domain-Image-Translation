param(
    [string]$Dataset = "apple2orange",
    [string]$Experiment = "apple2orange",
    [string]$ImageSize = "256",
    [string]$Device = "cpu"
)

python scripts/train.py --dataset-name $Dataset --experiment-name $Experiment --image-size $ImageSize --device $Device
