# ============================================================
# train.py
# Main script — ties everything together
# ============================================================

import torch
from src.model_loader import load_model_and_tokenizer
from src.lora_config import get_lora_config, apply_lora
from src.data_utils import load_and_prepare_dataset
from src.trainer import train_model

# ---- Configuration ----
# TinyLlama is ideal for Colab research:
# - 1.1B params (not 7B), fits comfortably in T4 with 4-bit
# - Trained on 3T tokens, genuinely capable
# - Instruction-tuned variant available for comparison
# Alternative: "mistralai/Mistral-7B-v0.1" (needs more VRAM, ~5-6GB)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./qlora-finetuned"

def main():
    print("=" * 60)
    print("QLoRA Fine-Tuning Pipeline")
    print("=" * 60)
    
    # 1. Load quantized model
    print("\n[1/4] Loading quantized model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # 2. Apply LoRA adapters
    print("\n[2/4] Applying LoRA adapters...")
    lora_config = get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    model = apply_lora(model, lora_config)
    
    # 3. Load and prepare dataset
    print("\n[3/4] Preparing dataset...")
    train_data, val_data = load_and_prepare_dataset(
        tokenizer=tokenizer,
        num_samples=5000,    # Reduce to 1000 for a quick test run
        max_length=512,
    )
    
    # 4. Train
    print("\n[4/4] Starting training...")
    trainer = train_model(model, tokenizer, train_data, val_data, OUTPUT_DIR)
    
    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
