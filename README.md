# QLoRA_Finetuning_Cheaply
# QLoRA Fine-Tuning: A Research Implementation

> Quantized Low-Rank Adaptation (QLoRA) fine-tuning of LLMs  
> on consumer hardware (Google Colab T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
This project implements QLoRA (Dettmers et al., 2023) from scratch using  
HuggingFace Transformers, PEFT, and bitsandbytes. It includes a full training  
pipeline, ablation experiments, and evaluation framework.

## Key Features
- 4-bit NF4 quantization via bitsandbytes
- LoRA adapter injection with PEFT
- Instruction tuning on Alpaca-cleaned dataset
- Runs on free Colab T4 GPU (~12GB VRAM)
- Ablation experiments: rank, alpha, target modules

## Quick Start
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/qlora-research
cd qlora-research
pip install -r requirements.txt
python train.py
\`\`\`

## Results
| Experiment | LoRA Rank | Perplexity (↓) | Train Loss |
|---|---|---|---|
| Baseline | r=16 | - | - |
| Low rank | r=4 | - | - |
| High rank | r=64 | - | - |

## Architecture
[Include your architecture diagram here]

## References
1. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs
2. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
3. Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers
