# ============================================================
# src/lora_config.py
# Defines which modules get LoRA adapters and the rank/alpha
# ============================================================

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
) -> LoraConfig:
    """
    Build the LoRA configuration.
    
    Key hyperparameters explained:
    
    r (rank): The bottleneck dimension of the adapter matrices.
      - Lower r (4, 8): fewer params, faster, possibly underfits
      - Higher r (32, 64): more expressive, more memory, can overfit
      - r=16 is a safe default for most tasks
    
    lora_alpha: Scaling factor for the adapter output.
      The effective learning rate of the adapter is lora_alpha/r.
      Convention: set alpha = 2*r or alpha = r.
      With alpha=32, r=16 → effective scale = 2.0
    
    lora_dropout: Regularization. 0.05–0.1 is typical.
    
    target_modules: Which weight matrices to apply LoRA to.
      In transformers, attention has q_proj, k_proj, v_proj, o_proj.
      The MLP has gate_proj, up_proj, down_proj.
      Applying to all attention + MLP modules is most powerful.
      For memory savings, apply only to q_proj and v_proj (standard LoRA).
    
    bias="none": Don't train bias terms (saves a small amount of memory).
    
    task_type=CAUSAL_LM: tells PEFT this is a language modeling task,
      which affects how the forward pass is set up.
    """
    if target_modules is None:
        # LLaMA / Mistral / TinyLlama architecture
        # These are the attention projection layers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj"         # MLP (include for full expressivity)
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model, lora_config: LoraConfig):
    """
    Wrap the quantized model with LoRA adapters.
    
    prepare_model_for_kbit_training() does several important things:
    1. Casts layer norms to FP32 (they must stay in high precision for stability)
    2. Freezes all base model parameters
    3. Enables gradient checkpointing hooks
    4. Casts the output embedding layer to FP32
    
    get_peft_model() then injects the A and B adapter matrices into
    every target module, keeping the original weights frozen.
    """
    # Step 1: Prepare quantized model for k-bit training
    # This MUST be called before get_peft_model for 4-bit models
    model = prepare_model_for_kbit_training(model)
    
    # Step 2: Inject LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Confirm what's trainable
    model.print_trainable_parameters()
    # Output should look like:
    # trainable params: 5,636,096 || all params: 1,105,199,104 || trainable%: 0.5100
    
    return model
