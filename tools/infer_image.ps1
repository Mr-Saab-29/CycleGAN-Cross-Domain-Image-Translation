param(
    [Parameter(Mandatory = $true)][string]$Input,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][ValidateSet("x2y", "y2x")][string]$Direction,
    [string]$Dataset = "apple2orange",
    [string]$Experiment = "apple2orange",
    [string]$ImageSize = "256",
    [string]$Device = "cpu"
)

python scripts/infer_image.py --dataset-name $Dataset --experiment-name $Experiment --image-size $ImageSize --device $Device --direction $Direction --input $Input --output $Output
