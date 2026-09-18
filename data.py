"""
Datasets for both training stages.

Two real-data hooks are marked TODO — that's where you plug in the actual
respiratory audio + symptom metadata once you have dataset access (UK
COVID-19, Coughvid, TBscreen, ICBHI, etc. — several need a data use
agreement, so start with whichever you can access first, e.g. Coughvid or
ICBHI which are the most openly downloadable).

Until then, `Dummy*Dataset` classes generate random-but-correctly-shaped
data so you can verify the whole pipeline (Stage 1 contrastive alignment,
Stage 2 LoRA fine-tuning) runs end-to-end before touching real data.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from configs import Config, TASK_PROMPTS

_SYMPTOM_BANK = [
    "fever", "dry cough", "wet cough", "fatigue", "shortness of breath",
    "chest tightness", "no smoking history", "smoker for 10 years",
    "no fever", "no other symptoms reported", "loss of taste and smell",
    "wheezing", "night sweats", "weight loss", "productive cough with blood",
]


def make_symptom_text(rng: random.Random, k: int = 3) -> str:
    """Mimic the paper's templated patient-metadata-to-text conversion."""
    chosen = rng.sample(_SYMPTOM_BANK, k=k)
    return "Patient reports: " + ", ".join(chosen) + "."


# ---------------------------------------------------------------------------
# Stage 1: audio <-> text contrastive pairs
# ---------------------------------------------------------------------------

@dataclass
class AudioTextPair:
    audio: torch.Tensor     # raw waveform (T,) at cfg.sample_rate, OR precomputed (audio_dim,) embedding
    text: str


class DummyAudioTextDataset(Dataset):
    """Random (waveform, symptom-text) pairs, correctly shaped for Stage 1.

    Swap this for a dataset that reads real audio files + real symptom
    metadata (see `RealAudioTextDataset` skeleton below) once you have data.
    """

    def __init__(self, cfg: Config, n_samples: int = 256, seed: int = 0):
        self.cfg = cfg
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.n_wave_samples = int(cfg.sample_rate * cfg.audio_seconds)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> AudioTextPair:
        waveform = torch.randn(self.n_wave_samples) * 0.05
        text = make_symptom_text(self.rng)
        return AudioTextPair(audio=waveform, text=text)


def collate_audio_text(batch: List[AudioTextPair]):
    audio = torch.stack([b.audio for b in batch], dim=0)
    texts = [b.text for b in batch]
    return audio, texts


# ---------------------------------------------------------------------------
# Stage 2: (audio, prompt, context, label) for classification fine-tuning
# ---------------------------------------------------------------------------

@dataclass
class ClassificationExample:
    audio: torch.Tensor
    prompt: str
    context: str
    label: int


class DummyClassificationDataset(Dataset):
    """Random examples shaped like Stage 2 inputs for one disease task."""

    def __init__(self, cfg: Config, task: str = "covid19", n_samples: int = 128, seed: int = 1):
        assert task in TASK_PROMPTS, f"unknown task {task}, pick from {list(TASK_PROMPTS)}"
        self.cfg = cfg
        self.task = task
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.n_wave_samples = int(cfg.sample_rate * cfg.audio_seconds)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> ClassificationExample:
        waveform = torch.randn(self.n_wave_samples) * 0.05
        context = make_symptom_text(self.rng)
        label = self.rng.randint(0, 1)
        return ClassificationExample(
            audio=waveform,
            prompt=TASK_PROMPTS[self.task],
            context=context,
            label=label,
        )


def collate_classification(batch: List[ClassificationExample]):
    audio = torch.stack([b.audio for b in batch], dim=0)
    prompts = [b.prompt for b in batch]
    contexts = [b.context for b in batch]
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    return audio, prompts, contexts, labels


# ---------------------------------------------------------------------------
# TODO: real-data skeletons
# ---------------------------------------------------------------------------
#
# class RealAudioTextDataset(Dataset):
#     """
#     index_csv columns expected: audio_path, symptom_text
#     (build symptom_text with the same templating logic as the paper's
#     Table 5 — one template per dataset, filled from patient metadata).
#     """
#     def __init__(self, cfg, index_csv):
#         import pandas as pd, torchaudio
#         self.cfg = cfg
#         self.df = pd.read_csv(index_csv)
#         self.torchaudio = torchaudio
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         wav, sr = self.torchaudio.load(row.audio_path)
#         if sr != self.cfg.sample_rate:
#             wav = self.torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
#         wav = wav.mean(dim=0)  # mono
#         target_len = int(self.cfg.sample_rate * self.cfg.audio_seconds)
#         if wav.numel() >= target_len:
#             wav = wav[:target_len]
#         else:
#             reps = target_len // wav.numel() + 1
#             wav = wav.repeat(reps)[:target_len]
#         return AudioTextPair(audio=wav, text=row.symptom_text)
