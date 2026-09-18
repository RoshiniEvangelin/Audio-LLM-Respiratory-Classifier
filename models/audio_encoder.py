"""
Audio encoder: CLAP (laion/clap-htsat-unfused), frozen, 512-dim output.

Swapped in for OPERA-CT because CLAP installs and runs with plain `pip
install transformers` — pure PyTorch, no conda environment, no shell
scripts, no separate repo, works the same on Windows/Mac/Linux. It's also a
defensible substitution on paper-fidelity grounds: RespiraMFM's own BTS
baseline uses CLAP for this exact role (frozen audio encoder feeding a
downstream classifier), and CLAP's audio embeddings already live in a space
aligned with text (it's a contrastive audio-text model itself), which is
arguably a *better* starting point for Stage 1's alignment objective than
OPERA-CT's audio-only embeddings were.

Important: CLAP expects 48kHz audio (not 16kHz) — `configs.Config.sample_rate`
defaults to 48000 to match. If you resample your own data pipeline
separately, make sure it lands on 48kHz before this encoder sees it.

Two options here:

1. `ClapAudioEncoder` — the real thing. Downloads ~600MB of pretrained
   weights from HuggingFace on first use (same mechanism Phi-2 used).
   Frozen; runs live in the training loop (no separate precompute step
   needed — CLAP is light enough to just run per-batch, unlike OPERA-CT's
   heavier offline-extraction-only workflow).

2. `MockAudioEncoder` — a tiny random-init CNN stand-in, for verifying
   pipeline shapes/wiring without a ~600MB download or GPU.
"""

import torch
import torch.nn as nn
import torchaudio


class MockAudioEncoder(nn.Module):
    """Random-init CNN over log-mel spectrograms -> out_dim embedding.

    NOT pretrained. Purely for validating tensor shapes / training loops
    before pulling down real CLAP weights.
    """

    def __init__(self, cfg, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or cfg.audio_dim
        self.cfg = cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=int(cfg.sample_rate * cfg.mel_win_ms / 1000),
            win_length=int(cfg.sample_rate * cfg.mel_win_ms / 1000),
            hop_length=int(cfg.sample_rate * cfg.mel_hop_ms / 1000),
            n_mels=cfg.n_mels,
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) -> (B, out_dim)"""
        mel = self.melspec(waveform)                      # (B, n_mels, frames)
        mel = torch.log(mel.clamp(min=1e-6)).unsqueeze(1)  # (B, 1, n_mels, frames)
        feat = self.net(mel).flatten(1)                    # (B, 64)
        return self.proj(feat)                              # (B, out_dim)


class ClapAudioEncoder(nn.Module):
    """Frozen pretrained CLAP audio tower -> 512-dim embedding.

    forward() accepts raw waveform batches at cfg.sample_rate (48000) as
    (B, T) float tensors — same call signature as MockAudioEncoder, so it's
    a drop-in swap everywhere else in the pipeline.
    """

    def __init__(self, cfg):
        super().__init__()
        from transformers import ClapModel, ClapFeatureExtractor

        self.cfg = cfg
        self.model_name = getattr(cfg, "clap_model_name", "laion/clap-htsat-unfused")
        self.feature_extractor = ClapFeatureExtractor.from_pretrained(self.model_name)
        self.model = ClapModel.from_pretrained(self.model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        assert self.feature_extractor.sampling_rate == cfg.sample_rate, (
            f"CLAP feature extractor expects {self.feature_extractor.sampling_rate}Hz "
            f"but cfg.sample_rate={cfg.sample_rate}. Set Config.sample_rate=48000."
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) float tensor at cfg.sample_rate -> (B, 512)"""
        device = waveform.device
        audio_list = [w.detach().cpu().numpy() for w in waveform]
        inputs = self.feature_extractor(
            audio_list, sampling_rate=self.cfg.sample_rate,
            return_tensors="pt", truncation="rand_trunc",
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.model.to(device)
        out = self.model.get_audio_features(**inputs)
        # NB: in current `transformers`, get_audio_features returns a
        # BaseModelOutputWithPooling whose .pooler_output is the actual
        # L2-normalized, projected 512-dim audio embedding (verified against
        # the installed transformers==5.17.0 source — older versions/docs
        # show it returning the embedding tensor directly, so if you're on
        # an older transformers pin and this errors, use `out` itself
        # instead of `out.pooler_output`).
        return out.pooler_output if hasattr(out, "pooler_output") else out


def build_audio_encoder(cfg, use_clap: bool = False):
    """Factory: real CLAP encoder if requested, mock stand-in otherwise."""
    if use_clap:
        print(f"[audio_encoder] loading CLAP ({getattr(cfg, 'clap_model_name', 'laion/clap-htsat-unfused')})"
              f" — downloads on first run, then cached locally.")
        return ClapAudioEncoder(cfg)
    print("[audio_encoder] using MockAudioEncoder (untrained, shape-only stand-in). "
          "Pass --use_clap for real pretrained audio features.")
    return MockAudioEncoder(cfg)
