"""
Real dataset loader for ICBHI 2017 (Respiratory Sound Database).

Expects the official ICBHI zip extracted somewhere with this layout
(exactly as the download provides it):

    ICBHI_final_database/
        101_1b1_Al_sc_Meditron.wav
        101_1b1_Al_sc_Meditron.txt   <- per-cycle annotations: start, end, crackles(0/1), wheezes(0/1)
        102_1b1_Ar_sc_Meditron.wav
        102_1b1_Ar_sc_Meditron.txt
        ...
    patient_diagnosis.csv            <- "patient_id,diagnosis" per line, no header
    demographic_info.txt             <- "patient_id age sex adult_BMI child_weight child_height", whitespace-sep

If your extraction put patient_diagnosis.csv / demographic_info.txt in a
different spot, just pass their paths explicitly — see `ICBHIIndex.__init__`.

Diagnosis labels in the official release: Healthy, COPD, URTI,
Bronchiectasis, Pneumonia, Bronchiolitis, LRTI, Asthma. RespiraMFM's ICBHI
task (T4) is framed as binary COPD-vs-not classification, which is the
default here (`target_diagnosis="COPD"`) — change it to build any of the
other one-vs-rest tasks the same dataset can support.

Context ("symptom") text is built from what ICBHI actually provides:
demographics (age, sex, BMI) plus the crackle/wheeze annotation summary for
that specific recording — NOT a fabricated symptom checklist. This is a
deliberately different (and arguably more honest) text signal than the
paper's UK-COVID-app-derived symptom text, since ICBHI simply doesn't
collect symptom checklists. Swap `build_context_text` if you want to try a
different templating.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from configs import Config, TASK_PROMPTS

# NOTE: these carry an `id` field (the recording's filename stem) alongside
# audio/text/label, unlike data.py's AudioTextPair/ClassificationExample.
# It's currently unused by the training scripts (CLAP runs live on raw audio,
# no by-id feature cache) but kept around since it's handy for logging/
# debugging which recording a batch element came from, and it's what you'd
# want if you ever add a precomputed-feature cache for speed later.


@dataclass
class ICBHIAudioTextItem:
    id: str
    audio: torch.Tensor
    text: str


@dataclass
class ICBHIClassificationItem:
    id: str
    audio: torch.Tensor
    prompt: str
    context: str
    label: int


def collate_icbhi_audio_text(batch: List[ICBHIAudioTextItem]):
    ids = [b.id for b in batch]
    audio = torch.stack([b.audio for b in batch], dim=0)
    texts = [b.text for b in batch]
    return ids, audio, texts


def collate_icbhi_classification(batch: List[ICBHIClassificationItem]):
    ids = [b.id for b in batch]
    audio = torch.stack([b.audio for b in batch], dim=0)
    prompts = [b.prompt for b in batch]
    contexts = [b.context for b in batch]
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    return ids, audio, prompts, contexts, labels


DIAGNOSIS_TASK_KEY = {
    "COPD": "copd",
    "Asthma": "asthma",
    "Pneumonia": "pneumonia",
    # URTI/LRTI/Bronchiectasis/Bronchiolitis have no matching TASK_PROMPTS
    # entry yet — add one to configs.TASK_PROMPTS if you target those.
}


def _read_diagnosis_csv(path: Path) -> Dict[str, str]:
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            patient_id, diagnosis = row[0].strip(), row[1].strip()
            mapping[patient_id] = diagnosis
    return mapping


def _read_demographics(path: Path) -> Dict[str, dict]:
    demo = {}
    if not path.exists():
        return demo
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = parts[0]
            demo[pid] = {
                "age": parts[1] if len(parts) > 1 else None,
                "sex": parts[2] if len(parts) > 2 else None,
            }
    return demo


def _read_cycle_annotations(txt_path: Path) -> Tuple[int, int, int]:
    """Returns (n_cycles, n_crackles, n_wheezes) for one recording's .txt file."""
    n_cycles = n_crackles = n_wheezes = 0
    if not txt_path.exists():
        return 0, 0, 0
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            n_cycles += 1
            n_crackles += int(float(parts[2]))
            n_wheezes += int(float(parts[3]))
    return n_cycles, n_crackles, n_wheezes


def build_context_text(age, sex, n_cycles, n_crackles, n_wheezes) -> str:
    bits = []
    if age:
        bits.append(f"age {age}")
    if sex:
        bits.append("male" if sex.upper().startswith("M") else "female" if sex.upper().startswith("F") else sex)
    demo_str = ", ".join(bits) if bits else "demographics unavailable"

    if n_cycles == 0:
        acoustic_str = "no cycle annotations available"
    else:
        crackle_str = f"crackles in {n_crackles}/{n_cycles} cycles" if n_crackles else "no crackles"
        wheeze_str = f"wheezes in {n_wheezes}/{n_cycles} cycles" if n_wheezes else "no wheezes"
        acoustic_str = f"{crackle_str}, {wheeze_str}"

    return f"Patient: {demo_str}. Recording findings: {acoustic_str}."


class ICBHISubset:
    """A read-only view over a subset of an ICBHIIndex's examples (e.g. the
    train or test half of a patient-level split). Shares the parent's prompt
    and audio-loading logic so it's a drop-in replacement for ICBHIIndex
    wherever one is expected (ICBHIAudioTextDataset / ICBHIClassificationDataset)."""

    def __init__(self, parent: "ICBHIIndex", examples: List[dict]):
        self.examples = examples
        self.prompt = parent.prompt
        self._parent = parent

    def __len__(self):
        return len(self.examples)

    def load_waveform(self, cfg: Config, audio_path: str) -> torch.Tensor:
        return self._parent.load_waveform(cfg, audio_path)


class ICBHIIndex:
    """Scans the ICBHI directory once and builds a list of examples shared
    by both the Stage-1 and Stage-2 dataset wrappers below."""

    def __init__(self, audio_dir: str, patient_diagnosis_csv: str,
                 demographic_info_txt: Optional[str] = None,
                 target_diagnosis: str = "COPD"):
        audio_dir = Path(audio_dir)
        diagnosis_map = _read_diagnosis_csv(Path(patient_diagnosis_csv))
        demo_map = _read_demographics(Path(demographic_info_txt)) if demographic_info_txt else {}

        if target_diagnosis not in DIAGNOSIS_TASK_KEY:
            raise ValueError(
                f"No TASK_PROMPTS entry for {target_diagnosis!r}. Add one to "
                f"configs.TASK_PROMPTS, or pick from {list(DIAGNOSIS_TASK_KEY)}."
            )
        self.prompt = TASK_PROMPTS[DIAGNOSIS_TASK_KEY[target_diagnosis]]

        self.examples: List[dict] = []
        wav_files = sorted(audio_dir.glob("*.wav"))
        skipped = 0
        for wav_path in wav_files:
            patient_id = wav_path.stem.split("_")[0]
            diagnosis = diagnosis_map.get(patient_id)
            if diagnosis is None:
                skipped += 1
                continue

            txt_path = wav_path.with_suffix(".txt")
            n_cycles, n_crackles, n_wheezes = _read_cycle_annotations(txt_path)
            demo = demo_map.get(patient_id, {})
            context = build_context_text(demo.get("age"), demo.get("sex"), n_cycles, n_crackles, n_wheezes)

            self.examples.append({
                "id": wav_path.stem,
                "audio_path": str(wav_path),
                "context": context,
                "label": 1 if diagnosis == target_diagnosis else 0,
            })

        if skipped:
            print(f"[ICBHIIndex] skipped {skipped} files with no matching patient_diagnosis.csv entry")
        print(f"[ICBHIIndex] indexed {len(self.examples)} recordings "
              f"({sum(e['label'] for e in self.examples)} positive for {target_diagnosis})")

    def __len__(self):
        return len(self.examples)

    def split_by_patient(self, test_frac: float = 0.2, seed: int = 42):
        """Split into (train, test) ICBHISubset objects by patient ID, so a
        given patient's recordings never appear in both splits — splitting by
        recording instead would leak the same patient's audio into both
        halves and inflate held-out accuracy/AUROC.

        Returns two ICBHISubset instances sharing this index's audio-loading
        logic; each behaves like an ICBHIIndex for dataset-building purposes.
        """
        patient_ids = sorted(set(e["id"].split("_")[0] for e in self.examples))
        rng = random.Random(seed)
        rng.shuffle(patient_ids)
        n_test = max(1, int(round(len(patient_ids) * test_frac)))
        test_patients = set(patient_ids[:n_test])

        train_examples = [e for e in self.examples if e["id"].split("_")[0] not in test_patients]
        test_examples = [e for e in self.examples if e["id"].split("_")[0] in test_patients]

        print(f"[ICBHIIndex] patient-level split: {len(patient_ids)} patients -> "
              f"{len(patient_ids) - n_test} train / {n_test} test patients "
              f"({len(train_examples)} train / {len(test_examples)} test recordings)")

        return ICBHISubset(self, train_examples), ICBHISubset(self, test_examples)

    def load_waveform(self, cfg: Config, audio_path: str) -> torch.Tensor:
        # soundfile instead of torchaudio.load: recent torchaudio versions
        # route .load()/.save() through an optional `torchcodec` dependency
        # that isn't installed by default (see requirements.txt) — soundfile
        # avoids that entirely and is just as easy on Windows.
        wav_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # (T, channels)
        wav = torch.from_numpy(wav_np.T)  # (channels, T)
        if sr != cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
        wav = wav.mean(dim=0)  # mono

        target_len = int(cfg.sample_rate * cfg.audio_seconds)
        if wav.numel() >= target_len:
            wav = wav[:target_len]
        else:
            reps = target_len // max(wav.numel(), 1) + 1
            wav = wav.repeat(reps)[:target_len]
        return wav


class ICBHIAudioTextDataset(Dataset):
    """Stage 1: (audio, context text) pairs from ICBHI — ignores labels."""

    def __init__(self, cfg: Config, index: ICBHIIndex):
        self.cfg = cfg
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> ICBHIAudioTextItem:
        ex = self.index.examples[idx]
        wav = self.index.load_waveform(self.cfg, ex["audio_path"])
        return ICBHIAudioTextItem(id=ex["id"], audio=wav, text=ex["context"])


class ICBHIClassificationDataset(Dataset):
    """Stage 2: (audio, prompt, context, label) examples from ICBHI."""

    def __init__(self, cfg: Config, index: ICBHIIndex):
        self.cfg = cfg
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> ICBHIClassificationItem:
        ex = self.index.examples[idx]
        wav = self.index.load_waveform(self.cfg, ex["audio_path"])
        return ICBHIClassificationItem(
            id=ex["id"], audio=wav, prompt=self.index.prompt, context=ex["context"], label=ex["label"],
        )