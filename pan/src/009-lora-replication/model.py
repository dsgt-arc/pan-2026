"""
Model creation and loading using the adapters library (AdapterHub).
- create_lora_model: for training (base model + fresh LoRA adapter)
- load_trained_model: for inference (base model + saved adapter)
"""

import adapters
from adapters import LoRAConfig
from transformers import AutoModelForSequenceClassification, AutoConfig

from config import (
    MODEL_NAME,
    NUM_LABELS,
    ADAPTER_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_ATTN_MATRICES,
)


def _load_base_model(model_name: str = MODEL_NAME):
    """Load the base Qwen2.5-1.5B model for sequence classification."""
    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = config.eos_token_id
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, torch_dtype="auto",
    )
    adapters.init(model)
    return model


def create_lora_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """
    Create a Qwen2.5-1.5B model with a fresh LoRA adapter for training.
    Uses the adapters library instead of PEFT.
    """
    model = _load_base_model(model_name)

    # Add LoRA adapter
    lora_config = LoRAConfig(
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        attn_matrices=LORA_ATTN_MATRICES,
    )
    model.add_adapter(ADAPTER_NAME, config=lora_config)
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)

    # Print parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    model.to(device)
    return model


def load_trained_model(
    adapter_path: str,
    model_name: str = MODEL_NAME,
    device: str = "cpu",
):
    """
    Load a saved LoRA adapter into the base model for inference.
    adapter_path can be a local directory or a HuggingFace repo ID.
    """
    model = _load_base_model(model_name)

    adapter_name = model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    model.to(device)
    model.eval()
    return model
