"""
Configuration for experiment 009-lora-replication.
All paths, model settings, LoRA hyperparameters, and training hyperparameters.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # pan-2026/
PAN_ROOT = PROJECT_ROOT / "pan"

DATA_DIR = PAN_ROOT / "src" / "pan-data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

EXTERNAL_DIR = DATA_DIR / "external"
HC3_FILE = EXTERNAL_DIR / "hc3_wiki_processed.jsonl"
RAID_FILE = EXTERNAL_DIR / "raid.jsonl"

EXPERIMENT_DIR = PAN_ROOT / "experiments" / "009-lora-replication"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"

SRC_003 = PAN_ROOT / "src" / "003-deberta"

# ── Model ──────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_LABELS = 2
MAX_LENGTH = 512

# ── LoRA (PEFT) ───────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ── Training ───────────────────────────────────────────────────
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2  # effective batch size = 16
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 50

# ── Chunking (for evaluation) ─────────────────────────────────
CHUNK_SIZE = 512
CHUNK_STRIDE = 256  # 50% overlap

# ── W&B ────────────────────────────────────────────────────────
WANDB_PROJECT = "pan-2026"
WANDB_ENTITY = "hersheys-baklava"
WANDB_RUN_NAME = "009-lora-replication"

# ── HuggingFace ────────────────────────────────────────────────
HF_REPO = "hersheys-baklava/qwen-lora-pan2026-009"
