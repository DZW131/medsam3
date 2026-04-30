# MedSAM3 Server Reproduction Notes

This note records the minimal server-side workflow for cloning this repository,
installing dependencies, downloading weights, and running inference.

## 1. Clone

```bash
git clone https://github.com/DZW131/medsam3.git
cd medsam3
```

## 2. Create Environment

Use a CUDA-capable server. The original project targets recent PyTorch/CUDA.
Python 3.12 is recommended by the bundled requirements.

```bash
conda create -n medsam3 python=3.12 -y
conda activate medsam3

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PyTorch installation fails or installs a CPU build, install the PyTorch wheel
that matches the server CUDA version first, then rerun the remaining installs.

## 3. Hugging Face Login

The base SAM3 checkpoint is downloaded from Hugging Face by the code path in
`sam3/model_builder.py`. Log in before first inference.

```bash
huggingface-cli login
```

If the CLI command is unavailable in your environment, try:

```bash
hf auth login
```

## 4. LoRA Weights

Download the SAM3 base checkpoint and MedSAM3 LoRA weights under ignored local
directories:

```bash
mkdir -p weights/sam3_base weights/medsam3_v1
# Put the SAM3 base checkpoint here:
#   weights/sam3_base/sam3.pt
# Put the released MedSAM3 LoRA file here:
#   weights/medsam3_v1/best_lora_weights.pt
```

This repository intentionally does not track `.pt`, `.pth`, `.ckpt`, or
`.safetensors` files.

## 5. Single Image Inference

Run inference with an explicit LoRA weight path:

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --sam3-checkpoint weights/sam3_base/sam3.pt \
  --weights weights/medsam3_v1/best_lora_weights.pt \
  --image path/to/image.jpg \
  --prompt "skin lesion" \
  --threshold 0.5 \
  --nms-iou 0.5 \
  --output outputs/demo_skin_lesion.png
```

For multiple prompts:

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --sam3-checkpoint weights/sam3_base/sam3.pt \
  --weights weights/medsam3_v1/best_lora_weights.pt \
  --image path/to/image.jpg \
  --prompt "skin lesion" "lesion" "abnormal skin region" \
  --output outputs/demo_multi_prompt.png
```

## 6. Training Data Layout

The native training script expects COCO-style split folders:

```text
data/
  train/
    _annotations.coco.json
    image_001.jpg
    ...
  valid/
    _annotations.coco.json
    image_101.jpg
    ...
```

Then set `training.data_dir` in `configs/full_lora_config.yaml`, or create a
new config file, and run:

```bash
python train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

For a 24 GB GPU, start with a light config, `batch_size=1`, mixed precision, and
gradient accumulation.
