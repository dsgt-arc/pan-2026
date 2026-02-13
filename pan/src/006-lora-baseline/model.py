"""
LoRA model creation for Qwen2.5-1.5B sequence classification.
Wraps the causal LM with PEFT LoRA adapters for binary classification.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

from config import (
    MODEL_NAME,
    NUM_LABELS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


def create_lora_model(device: str = "cpu"):
    """
    Creates a Qwen2.5-1.5B model wrapped with LoRA adapters for
    sequence classification.

    AutoModelForSequenceClassification will add a classification head
    on top of the causal LM and use the last non-padding token's
    representation for classification.

    Returns:
        peft_model  – the PEFT-wrapped model
    """
    # 1. Load base model for sequence classification
    print(f"Loading base model: {MODEL_NAME}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.pad_token_id = config.eos_token_id  # Qwen uses eos as pad
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype="auto",
    )

    # 2. LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # 3. Wrap with PEFT
    peft_model = get_peft_model(base_model, lora_config)

    # 4. Print parameter summary
    peft_model.print_trainable_parameters()

    peft_model.to(device)
    return peft_model
