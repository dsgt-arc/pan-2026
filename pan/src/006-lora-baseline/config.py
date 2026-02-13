"""
Configuration for experiment 006-lora-baseline.
Central place for paths, model settings, and LoRA hyperparameters.

Uses Qwen2.5-1.5B as the base LLM with LoRA adapters for
sequence classification on AI-generated text detection.
"""

from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # pan-2026/
PAN_ROOT = PROJECT_ROOT / "pan"

DATA_DIR = PAN_ROOT / "src" / "pan-data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
HC3_FILE = DATA_DIR / "external" / "hc3_wiki_processed.jsonl"

EXPERIMENT_DIR = PAN_ROOT / "experiments" / "006-lora-baseline"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"

SRC_003 = PAN_ROOT / "src" / "003-deberta"

# ──────────────────────────── Model ────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_LABELS = 2
MAX_LENGTH = 512

# ──────────────────────────── LoRA ─────────────────────────────
LORA_R = 16            # Low-rank dimension
LORA_ALPHA = 32        # Scaling factor (alpha/r = 2)
LORA_DROPOUT = 0.1     # Dropout on LoRA layers
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Qwen attention projections
LORA_TASK_TYPE = "SEQ_CLS"

# ──────────────────────────── Training ─────────────────────────
LEARNING_RATE = 2e-4       # Higher than full fine-tuning; recommended for LoRA
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4  # Effective batch size = 16
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

# ──────────────────────────── Chunking ─────────────────────────
CHUNK_SIZE = 512
CHUNK_STRIDE = 256  # 50 % overlap, same as 003
