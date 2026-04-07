PYTHON ?= python
UVICORN ?= $(PYTHON) -m uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
DATASET ?= apple2orange
EXPERIMENT ?= apple2orange
IMAGE_SIZE ?= 256
DEVICE ?= cpu

.PHONY: help install api train evaluate infer-sample infer-image

help:
	@echo "Targets:"
	@echo "  make install"
	@echo "  make api"
	@echo "  make train"
	@echo "  make evaluate"
	@echo "  make infer-sample"
	@echo "  make infer-image INPUT=path/to/image.jpg OUTPUT=translated.png DIRECTION=x2y"

install:
	$(PYTHON) -m pip install -r requirements.txt

api:
	set MODEL_DATASET_NAME=$(DATASET) && \
	set MODEL_EXPERIMENT_NAME=$(EXPERIMENT) && \
	set MODEL_CHECKPOINT_ROOT=checkpoints/pytorch && \
	set MODEL_OUTPUT_ROOT=outputs && \
	set MODEL_IMAGE_SIZE=$(IMAGE_SIZE) && \
	set MODEL_DEVICE=$(DEVICE) && \
	$(UVICORN) app.main:app --host $(HOST) --port $(PORT)

train:
	$(PYTHON) scripts/train.py --dataset-name $(DATASET) --experiment-name $(EXPERIMENT) --image-size $(IMAGE_SIZE) --device $(DEVICE)

evaluate:
	$(PYTHON) scripts/evaluate.py --dataset-name $(DATASET) --experiment-name $(EXPERIMENT) --image-size $(IMAGE_SIZE) --device $(DEVICE)

infer-sample:
	$(PYTHON) scripts/generate_samples.py --dataset-name $(DATASET) --experiment-name $(EXPERIMENT) --image-size $(IMAGE_SIZE) --device $(DEVICE)

infer-image:
	$(PYTHON) scripts/infer_image.py --dataset-name $(DATASET) --experiment-name $(EXPERIMENT) --image-size $(IMAGE_SIZE) --device $(DEVICE) --direction $(DIRECTION) --input $(INPUT) --output $(OUTPUT)
