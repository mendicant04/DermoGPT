# 🩺 DermoGPT

**DermoGPT** is a dermatology vision-language modeling project for morphology-first clinical reasoning. The release contains training, inference, and evaluation code for Qwen-VL-style multimodal models, together with DermoBench benchmark metadata and prompt documentation.

> This codebase is built on top of [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune). We thank the original authors for their excellent open-source fine-tuning framework.

### DermoGPT is available at https://huggingface.co/mendicant04/DermoGPT-RL
### DermoBench is available at https://huggingface.co/datasets/mendicant04/DermoBench
### DermoInstruct is available at https://huggingface.co/datasets/mendicant04/DermoInstruct

## ✨ Highlights

- 🧬 **Morphology-first reasoning**: supports outputs grounded in standardized dermatology morphology.
- 🔍 **DermoBench evaluation**: covers morphology description, morphology bottlenecks, diagnosis MCQA, hierarchical diagnosis, reasoning, and fairness evaluation.
- 🧠 **Qwen-VL fine-tuning stack**: supports SFT, LoRA/QLoRA, GRPO-style training utilities, and LoRA weight merging.
- 🚀 **Flexible inference**: supports local checkpoints, LoRA checkpoints, and OpenAI-compatible API backends.
- 📋 **Prompt transparency**: all major benchmark, synthesis, inference, and judge prompts are collected in [prompts.md](prompts.md).

## 📦 Repository Layout

```text
.
├── dataset_final/
│   └── benchmark/              # Benchmark metadata, MCQA JSON files, and evaluation scripts
├── scripts/                    # Training, inference, and weight-merging wrappers
├── src/
│   ├── dataset/                # SFT/DPO/GRPO datasets
│   ├── infer/                  # Local, LoRA, and API inference code
│   ├── loss/                   # Classification losses
│   ├── model/                  # Classification model utilities
│   ├── train/                  # Training entrypoints and reward functions
│   └── trainer/                # SFT/DPO/GRPO trainers
├── environment.yaml
├── requirements.txt
├── prompts.md
└── README.md
```


## 🧪 DermoBench Tasks

DermoBench evaluates four clinical axes:

| Axis | Tasks | Description |
|---|---|---|
| Morphology | Task 1.1-1.4 | Open-ended morphology, morph-grounded description, Derm7pt MCQA, SkinCon MCQA |
| Diagnosis | Task 2.1-2.3 | 4/25-choice diagnosis, hierarchical diagnosis, modality-focused diagnosis |
| Reasoning | Task 3.1-3.2 | CoT diagnosis and morphology-grounded reasoning |
| Fairness | Task 4 | DDI-based skin-tone fairness evaluation |

The benchmark files included here are derived artifacts and metadata. Original images should be obtained from the corresponding public dataset providers according to their licenses.

## ⚙️ Installation

Recommended environment:

- Python 3.12
- PyTorch with CUDA
- `transformers`
- `qwen-vl-utils`
- `flash-attn` installed after the base environment

Using `requirements.txt`:

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

Using `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

## 🏋️ Training

Full fine-tuning:

```bash
TRAIN_DATA_PATH=/path/to/train.json \
IMAGE_FOLDER=/path/to/images \
bash scripts/finetune.sh
```

LoRA fine-tuning:

```bash
TRAIN_DATA_PATH=/path/to/train.json \
IMAGE_FOLDER=/path/to/images \
bash scripts/finetune_lora.sh
```

Vision LoRA fine-tuning:

```bash
TRAIN_DATA_PATH=/path/to/train.json \
IMAGE_FOLDER=/path/to/images \
bash scripts/finetune_lora_vision.sh
```

Merge LoRA weights:

```bash
SOURCE_MODEL_PATH=/path/to/lora_checkpoint \
MERGED_MODEL_PATH=/path/to/merged_model \
bash scripts/merge_lora_weights.sh
```

## 🔬 Inference

Set `DATASET_ROOT` to the root containing `dataset_final`-style image paths.

### Local Baseline

```bash
MODEL_PATH=/path/to/model \
MODEL_FAMILY=qwen3-vl-8b \
DATASET_ROOT=dataset_final \
DEVICE=cuda:0 \
bash scripts/infer_baseline_local.sh all
```

### API Backend

```bash
BASE_URL=https://your-api-endpoint/v1 \
API_KEY=your_api_key \
API_MODEL=your_model_name \
DATASET_ROOT=dataset_final \
NUM_WORKERS=4 \
bash scripts/infer_baseline_api.sh all
```

Task-level runs are also supported:

```bash
bash scripts/infer_baseline_local.sh 1.1 1.2 3.1 3.2
bash scripts/infer_lora.sh 2.1
bash scripts/infer_baseline_api.sh 4.1
```

## 📊 Evaluation

Open-ended morphology and reasoning tasks use LLM-as-a-Judge scripts:

```bash
python dataset_final/benchmark/task1/1_1_description_wo_morph/eval_task1_1_gemini_2-5_pro_judge.py
python dataset_final/benchmark/task1/1_2_description_w_morph/eval_task1_2_gemini_2-5_pro_judge.py
python dataset_final/benchmark/task3/3_1/eval_task3_1_gemini_2-5_pro_judge.py
python dataset_final/benchmark/task3/3_2/eval_task3_2_gemini_2-5_pro_judge.py
```

MCQA and fairness evaluation utilities are included under:

```text
src/infer/mcq/
dataset_final/benchmark/task4/
```

Prompt details for model generation and judging are documented in [prompts.md](prompts.md).

## 📁 Data Release Notes

- This repository is intended for code, prompts, benchmark metadata, and evaluation utilities.
- Raw dermatology images are not redistributed here.
- Users should download original images from the source datasets and follow their licenses and intended-use policies.

## ⚠️ Medical Disclaimer

DermoGPT is a research project. It is not a medical device and should not be used as a substitute for professional clinical judgment, diagnosis, or treatment.

## 🙏 Acknowledgements

This project builds on:

- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL model families
- Public dermatology datasets used to construct DermoInstruct and DermoBench

## 📚 Citation

If you find this repository useful, please cite the DermoGPT paper.
