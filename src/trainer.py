# ============================================================
# src/trainer.py
# Full training loop using HuggingFace Trainer + TRL's SFTTrainer
# ============================================================

from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import os

def get_training_args(output_dir: str = "./results") -> TrainingArguments:
    """
    Training arguments tuned for T4 (12GB VRAM).
    
    Batch size strategy:
    - per_device_train_batch_size=1: Only 1 sample per GPU step
    - gradient_accumulation_steps=4: Accumulate gradients over 4 steps
    → Effective batch size = 1 × 4 = 4
    This simulates a batch of 4 while using memory of batch size 1.
    
    Learning rate:
    - 2e-4 is standard for LoRA adapters (higher than full FT)
    - LoRA adapters start from zero and need to learn quickly
    - Cosine scheduler decays smoothly from lr → 0
    
    Saving strategy:
    - save_steps=50: Save checkpoint every 50 steps
    - load_best_model_at_end: Restore best checkpoint after training
    
    fp16=True: Mixed precision training. Non-quantized tensors use FP16.
    Works on T4. (Use bf16=True on A100/H100 for better stability.)
    
    max_grad_norm=0.3: Gradient clipping prevents exploding gradients.
    Especially important with QLoRA where gradients can be noisy.
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Batch & accumulation (tuned for T4 12GB)
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,       # effective batch = 4
        
        # Learning rate schedule
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,                   # 3% of steps for warmup
        
        # Memory optimizations
        fp16=True,                            # mixed precision (T4)
        gradient_checkpointing=True,          # recompute activations
        optim="paged_adamw_32bit",            # paged optimizer (QLoRA paper)
        
        # Regularization
        weight_decay=0.001,
        max_grad_norm=0.3,                    # gradient clipping
        
        # Logging & saving
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=3,                  # keep only 3 checkpoints
        load_best_model_at_end=True,
        
        # Reporting
        report_to="none",                    # change to "wandb" if tracking
        seed=42,
    )


def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./results"):
    """
    Run the QLoRA fine-tuning loop.
    
    We use TRL's SFTTrainer (Supervised Fine-Tuning Trainer) which:
    - Handles packing multiple short sequences into one for efficiency
    - Provides cleaner API for instruction tuning
    - Compatible with PEFT LoRA models out of the box
    
    The packing=False setting is important: with max_length=512 and
    diverse instruction lengths, we pad each sample individually.
    Setting packing=True would concatenate samples (slightly faster
    but can confuse the model about example boundaries).
    """
    training_args = get_training_args(output_dir)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",           # column name from our dataset
        max_seq_length=512,
        packing=False,
        args=training_args,
    )
    
    # Track memory before training
    before_mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory before training: {before_mem:.2f} GB")
    
    # ---- TRAIN ----
    trainer.train()
    
    after_mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after training: {after_mem:.2f} GB")
    
    # Save the final adapter weights
    # This saves ONLY the LoRA adapter (~10-40MB), not the full model!
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")
    
    return trainer
