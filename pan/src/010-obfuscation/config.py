"""
Configuration for experiment 010-obfuscation.
All paths, model settings, LoRA hyperparameters, training hyperparameters,
and augmentation parameters.
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
MAGE_FILE = EXTERNAL_DIR / "mage.jsonl"
OPENGPTTEXT_FILE = EXTERNAL_DIR / "opengpttext.jsonl"

EXPERIMENT_DIR = PAN_ROOT / "experiments" / "010-obfuscation"
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

# ── Augmentation ───────────────────────────────────────────────
HOMOGLYPH_FRAC = 0.10      # fraction of AI texts to apply homoglyph augmentation
HOMOGLYPH_PROB = 0.05      # per-character homoglyph swap probability
ZWJ_PROB = 0.05            # per-character zero-width joiner insertion probability
SYNONYM_FRAC = 0.10        # fraction of AI texts to apply synonym replacement
SYNONYM_PROB = 0.20        # per-word synonym replacement probability

# ── Chunking (for evaluation) ─────────────────────────────────
CHUNK_SIZE = 512
CHUNK_STRIDE = 256  # 50% overlap

# ── W&B ────────────────────────────────────────────────────────
WANDB_PROJECT = "pan-2026"
WANDB_ENTITY = "spar"
WANDB_RUN_NAME = "010-obfuscation"

# ── HuggingFace ────────────────────────────────────────────────
HF_REPO = "hersheys-baklava/qwen-lora-pan2026-010-obfuscation"
