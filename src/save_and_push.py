# ============================================================
# src/save_and_push.py
# Save locally and optionally push to Hugging Face Hub
# ============================================================

from peft import PeftModel
import torch

def save_adapter(model, tokenizer, save_path: str):
    """
    Save only the LoRA adapter weights (~10–50MB).
    The base model is NOT saved — only the small delta.
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Adapter saved to: {save_path}")
    # Files created:
    # adapter_config.json  — LoRA hyperparams (r, alpha, modules)
    # adapter_model.safetensors — the actual A and B matrices


def merge_and_save(model, tokenizer, save_path: str):
    """
    Merge LoRA weights into the base model and save the full model.
    
    Use this for deployment when you want faster inference
    (no separate adapter loading step).
    
    NOTE: merged model is in FP16 (~2.5GB for 1.1B, ~13GB for 7B).
    Only do this if you have enough disk/RAM.
    """
    # Merge adapter weights into base model
    merged_model = model.merge_and_unload()
    
    # Save in half precision to save disk
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"Merged model saved to: {save_path}")


def push_to_hub(model, tokenizer, repo_name: str):
    """
    Upload adapter to Hugging Face Hub.
    Others can load it with: PeftModel.from_pretrained(base, your_repo)
    """
    model.push_to_hub(repo_name, use_auth_token=True)
    tokenizer.push_to_hub(repo_name, use_auth_token=True)
    print(f"Pushed to: https://huggingface.co/{repo_name}")
