# ── Stage 1: download model weights ──────────────────────────────
FROM python:3.11-slim AS downloader

RUN pip install --no-cache-dir huggingface_hub transformers peft torch

# Download base model and LoRA adapter at build time so the image is
# fully self-contained (TIRA runs sandboxed with no internet).
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig; \
from peft import PeftModel; \
name = 'Qwen/Qwen2.5-1.5B'; \
cfg = AutoConfig.from_pretrained(name); \
cfg.pad_token_id = cfg.eos_token_id; \
cfg.num_labels = 2; \
cfg.problem_type = 'single_label_classification'; \
m = AutoModelForSequenceClassification.from_pretrained(name, config=cfg); \
m = PeftModel.from_pretrained(m, 'hersheys-baklava/qwen-lora-pan2026-010-obfuscation'); \
m = m.merge_and_unload(); \
m.save_pretrained('/model'); \
t = AutoTokenizer.from_pretrained(name); \
t.save_pretrained('/model'); \
"

# ── Stage 2: runtime image ───────────────────────────────────────
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch \
    transformers \
    numpy \
    tqdm

COPY --from=downloader /model /model
COPY run_submission.py /app/run_submission.py

WORKDIR /app

ENTRYPOINT ["python", "run_submission.py"]
