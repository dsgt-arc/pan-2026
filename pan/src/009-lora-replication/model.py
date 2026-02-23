"""
Model creation and loading using PEFT (LoRA).
- create_lora_model: for training (base model + fresh LoRA adapters)
- load_trained_model: for inference (base model + merge saved adapter)
"""

from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from config import (
    MODEL_NAME,
    NUM_LABELS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


def create_lora_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """
    Create a Qwen2.5-1.5B model with LoRA adapters for training.
    Returns the PEFT-wrapped model.
    """
    import torch
    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = config.eos_token_id
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, torch_dtype=torch.bfloat16,
    )

    # Gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.to(device)
    return peft_model


def load_trained_model(
    adapter_path: str,
    model_name: str = MODEL_NAME,
    device: str = "cpu",
):
    """
    Load a saved LoRA adapter and merge it into the base model for inference.
    adapter_path can be a local directory or a HuggingFace repo ID.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = config.eos_token_id
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, torch_dtype="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # merge for faster inference
    model.to(device)
    model.eval()
    return model
