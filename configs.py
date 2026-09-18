"""
Hyperparameters transcribed from the RespiraMFM paper (arXiv:2606.09966).

Where the paper doesn't state a value explicitly, a sensible default is used
and flagged with a comment. Everything here is a plain dataclass so you can
override individual fields from a training script or CLI without touching
this file, e.g.:

    from configs import Config
    cfg = Config(llm_name="microsoft/phi-2", batch_size=4)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ---- Audio encoder (CLAP, swapped in for the paper's OPERA-CT) ----
    audio_dim: int = 512           # CLAP (laion/clap-htsat-unfused) projection_dim
    clap_model_name: str = "laion/clap-htsat-unfused"
    sample_rate: int = 48000       # CLAP expects 48kHz audio (not OPERA-CT's 16kHz)
    audio_seconds: float = 8.0     # fixed clip length (truncate/pad)
    mel_win_ms: float = 64.0       # only used by MockAudioEncoder's mel-spectrogram
    mel_hop_ms: float = 32.0       # only used by MockAudioEncoder's mel-spectrogram
    n_mels: int = 64               # only used by MockAudioEncoder's mel-spectrogram

    # ---- LLM backbone / text encoder ----
    # Paper's best-performing LLM. Swap to something smaller
    # (e.g. "microsoft/phi-1_5" ~1.3B, "gpt2" for pure smoke-testing) if a
    # single GPU can't fit Phi-2 + LoRA + activations at batch_size=16.
    llm_name: str = "microsoft/phi-2"
    llm_hidden_dim: int = 2560     # Phi-2 hidden size (d in the paper)
    max_seq_len: int = 256

    # ---- Stage 1: contrastive alignment projection head ----
    proj_hidden_dim: int = 1024
    proj_dropout: float = 0.1
    temperature: float = 0.07
    stage1_epochs: int = 500
    stage1_lr: float = 1e-3
    stage1_batch_size: int = 16    # not stated in paper; inferred from Stage 2
    stage1_optimizer: str = "adamw"

    # ---- Stage 2: LoRA instruction tuning ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "dense")  # Phi-2 attn proj names
    stage2_epochs: int = 20
    stage2_batch_size: int = 16
    stage2_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.06     # linear warmup; paper says "linear warmup", ratio not stated

    # ---- Task ----
    num_classes: int = 2           # binary disease classification per task

    # ---- Misc ----
    seed: int = 42
    device: str = "cuda"           # falls back to cpu automatically if unavailable
    use_4bit: bool = False         # set True to load the LLM in 4-bit (bitsandbytes) on small GPUs
    checkpoint_dir: str = "./checkpoints"


TASK_PROMPTS = {
    # Minimal examples mirroring the paper's Table-style task prompts.
    # Extend/replace per dataset when you wire in real data.
    "covid19": "Classify whether the participant has COVID-19 based on the "
                "respiratory sound and the patient information below.",
    "tuberculosis": "Classify whether the participant has tuberculosis based on "
                     "the respiratory sound and the patient information below.",
    "copd": "Classify whether the participant has COPD based on the "
            "respiratory sound and the patient information below.",
    "asthma": "Classify whether the participant has asthma based on the "
              "respiratory sound and the patient information below.",
    "pneumonia": "Classify whether the participant has pneumonia based on the "
                 "respiratory sound and the patient information below.",
}