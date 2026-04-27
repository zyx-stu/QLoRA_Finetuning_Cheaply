# ============================================================
# src/data_utils.py
# Dataset loading and prompt formatting for instruction tuning
# ============================================================

from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict

# ---- Prompt template ----
# We use the Alpaca instruction format — widely used for instruction tuning.
# The model learns to follow the pattern: instruction → response

ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# End-of-sequence token appended to each training example
EOS_TOKEN = None  # Will be set from tokenizer


def format_alpaca_sample(sample: Dict) -> Dict:
    """
    Format a single dataset sample into the instruction-tuning template.
    We add the EOS token so the model learns when to stop generating.
    """
    text = ALPACA_PROMPT.format(
        instruction=sample.get("instruction", ""),
        input=sample.get("input", ""),
        output=sample.get("output", "")
    ) + EOS_TOKEN
    return {"text": text}


def load_and_prepare_dataset(
    dataset_name: str = "yahma/alpaca-cleaned",
    tokenizer: PreTrainedTokenizer = None,
    max_length: int = 512,
    num_samples: int = 5000,   # Use subset for Colab speed
    val_split: float = 0.1,
):
    """
    Load, format, and tokenize the dataset.
    
    Why yahma/alpaca-cleaned?
    - 51K instruction-following samples cleaned from the original Stanford Alpaca
    - Small enough to train on in Colab (3–5 hours for 5K samples on T4)
    - Diverse enough to demonstrate generalization
    - No harmful content (cleaned)
    
    max_length=512 is the sweet spot for T4:
    - Shorter = faster, less memory, less context
    - Longer = richer context but quadratic attention memory growth
    - 512 fits well within T4 12GB at batch_size=1 with 4-bit model
    
    Tokenization with truncation+padding:
    - We pack everything to max_length for uniform batch shapes
    - padding='max_length' makes all sequences the same length in a batch
    - truncation=True drops tokens beyond max_length
    """
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Use a subset to fit Colab time constraints
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format into instruction template
    dataset = dataset.map(format_alpaca_sample)
    
    # Tokenize
    def tokenize(sample):
        """
        Tokenize and create labels.
        
        Important: For causal LM training, labels = input_ids.
        The loss is computed on all tokens (we're teaching the full format).
        Advanced technique: mask the instruction tokens and only compute loss
        on the response — this focuses learning on output quality.
        """
        result = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Clone input_ids as labels; pad tokens get label -100 (ignored in loss)
        result["labels"] = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in result["input_ids"]
        ]
        return result
    
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    
    # Train/validation split
    split = dataset.train_test_split(test_size=val_split, seed=42)
    
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    print(f"Sample input length: {sum(1 for x in split['train'][0]['input_ids'] if x != tokenizer.pad_token_id)} tokens")
    
    return split['train'], split['test']
