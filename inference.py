"""
Minimal end-to-end inference example: one real audio clip + patient context
-> class probabilities.

Run on a real ICBHI recording (matching the COPD classifier you trained):
    python inference.py --use_clap --task copd \
        --audio_path "C:\\path\\to\\some_recording.wav" \
        --stage1_ckpt ./checkpoints/stage1_projection.pt \
        --stage2_ckpt ./checkpoints/stage2_model.pt

If --audio_path is omitted, falls back to random dummy audio (a wiring
smoke test only, not a real prediction).
"""

import argparse

import torch
import torch.nn.functional as F

from configs import Config, TASK_PROMPTS
from models.audio_encoder import build_audio_encoder
from models.llm_backbone import RespiraMFM


def load_real_waveform(cfg, audio_path):
    import soundfile as sf
    import torchaudio

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
    return wav.unsqueeze(0)  # (1, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="copd")
    ap.add_argument("--stage1_ckpt", type=str, default="./checkpoints/stage1_projection.pt")
    ap.add_argument("--stage2_ckpt", type=str, default="./checkpoints/stage2_model.pt")
    ap.add_argument("--use_clap", action="store_true",
                     help="use real pretrained CLAP audio features instead of MockAudioEncoder")
    ap.add_argument("--audio_path", type=str, default=None,
                     help="path to a real .wav file; omit to use random dummy audio")
    ap.add_argument("--context", type=str,
                     default="Patient: demographics unavailable. Recording findings: no cycle annotations available.",
                     help="context text describing the patient/recording (see data_icbhi.py::build_context_text "
                          "for the same templating used during training)")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    audio_encoder = build_audio_encoder(cfg, use_clap=args.use_clap)
    model = RespiraMFM(cfg)
    model.load_stage1_projection(args.stage1_ckpt, freeze=True)

    ckpt = torch.load(args.stage2_ckpt, map_location=device)
    model.lm.load_state_dict(ckpt["lora"], strict=False)
    model.classifier.load_state_dict(ckpt["classifier"])
    model.lm.eval()

    if args.audio_path:
        waveform = load_real_waveform(cfg, args.audio_path)
        print(f"[inference] using real audio: {args.audio_path}")
    else:
        waveform = torch.randn(1, int(cfg.sample_rate * cfg.audio_seconds)) * 0.05
        print("[inference] no --audio_path given -- using random dummy audio "
              "(wiring check only, not a real prediction)")

    context_text = args.context

    with torch.no_grad():
        audio_emb = audio_encoder(waveform).float()
        logits = model(audio_emb, [TASK_PROMPTS[args.task]], [context_text])
        probs = F.softmax(logits.float(), dim=-1)[0]

    print(f"task={args.task}")
    print(f"context={context_text!r}")
    print(f"P(negative)={probs[0].item():.3f}  P(positive)={probs[1].item():.3f}")
    print(f"predicted: {'POSITIVE' if probs[1] > probs[0] else 'NEGATIVE'}")


if __name__ == "__main__":
    main()