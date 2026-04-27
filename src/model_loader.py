# ============================================================
# src/model_loader.py
# Loads a quantized base model ready for QLoRA fine-tuning
# ============================================================

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

def get_bnb_config() -> BitsAndBytesConfig:
    """
    Build the 4-bit quantization configuration.
    
    NF4 (NormalFloat4) is the key format from the QLoRA paper.
    It maps float values to a grid of 16 values chosen to be 
    optimal for normally-distributed weights.
    
    bnb_4bit_use_double_quant=True enables Double Quantization:
    the quantization constants themselves are quantized (FP32→FP8),
    saving ~0.37 bits/parameter additionally.
    
    compute_dtype=float16 means that during the forward pass,
    weights are dequantized to FP16 for matrix multiplication.
    (Use bfloat16 on A100/H100; T4 doesn't support it natively.)
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,                        # 4-bit quantization
        bnb_4bit_quant_type="nf4",                # NormalFloat4 format
        bnb_4bit_use_double_quant=True,            # Double quantization
        bnb_4bit_compute_dtype=torch.float16,      # Compute dtype (T4: fp16)
    )


def load_model_and_tokenizer(model_name: str, device_map: str = "auto"):
    """
    Load a quantized causal LM and its tokenizer.
    
    Args:
        model_name: HuggingFace model ID, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device_map: "auto" lets accelerate handle multi-GPU or CPU offload
    
    Returns:
        model, tokenizer
    """
    bnb_config = get_bnb_config()
    
    # ---- Load the model ----
    # This downloads ~600MB for TinyLlama (vs ~13GB for LLaMA-2-7B in FP16)
    # The model weights are stored in 4-bit but dequantized on each forward pass
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,            # needed for some models (Falcon, etc.)
        torch_dtype=torch.float16,         # dtype for non-quantized tensors
    )
    
    # Disable the model's built-in cache during training — it's incompatible
    # with gradient checkpointing and wastes memory
    model.config.use_cache = False
    
    # Enable gradient checkpointing: instead of storing all intermediate
    # activations for backprop, recompute them during the backward pass.
    # This trades ~30% extra compute for ~40% memory savings. Essential on T4.
    model.gradient_checkpointing_enable()
    
    # ---- Load the tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Most causal LMs don't have a pad token; we use EOS as pad.
    # This is important: without a pad token, batched training fails.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Pad on the right for causal LMs (left-padding causes attention mask issues)
    tokenizer.padding_side = "right"
    
    print(f"Model loaded. Trainable params: {count_trainable_params(model)}")
    return model, tokenizer


def count_trainable_params(model) -> str:
    """Human-readable trainable parameter count."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
