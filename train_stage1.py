"""
Stage 1: contrastive audio-language alignment.

Trains only the projection head f_theta (audio_dim -> 1024 -> 2560) so that
CLAP audio embeddings and frozen-Phi-2 text embeddings for the SAME
patient land close together in the shared 2560-dim space, and embeddings
for different patients land far apart (InfoNCE / CLIP-style loss).

Run:
    python train_stage1.py --epochs 5 --n_samples 64   # quick smoke test, mock audio encoder
    python train_stage1.py --use_clap                   # real CLAP features (downloads ~600MB first run)
    python train_stage1.py --use_clap --icbhi_audio_dir path/to/ICBHI_final_database  # real data + real audio
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config
from data import DummyAudioTextDataset, collate_audio_text
from losses import clip_contrastive_loss
from models.audio_encoder import build_audio_encoder
from models.llm_backbone import FrozenTextEncoder
from models.projection import ProjectionHead
from utils import set_seed, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--use_clap", action="store_true",
                     help="use real pretrained CLAP audio features instead of MockAudioEncoder")
    ap.add_argument("--out", type=str, default="./checkpoints/stage1_projection.pt")
    # Real-data flags: pass --icbhi_audio_dir to train on ICBHI instead of dummy data.
    ap.add_argument("--icbhi_audio_dir", type=str, default=None)
    ap.add_argument("--icbhi_diagnosis_csv", type=str, default=None)
    ap.add_argument("--icbhi_demographics", type=str, default=None)
    ap.add_argument("--icbhi_target_diagnosis", type=str, default="COPD")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.stage1_epochs = args.epochs
    if args.batch_size is not None:
        cfg.stage1_batch_size = args.batch_size

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[stage1] device={device}")

    audio_encoder = build_audio_encoder(cfg, use_clap=args.use_clap)

    if args.icbhi_audio_dir:
        from pathlib import Path as _Path
        from data_icbhi import ICBHIIndex, ICBHIAudioTextDataset, collate_icbhi_audio_text
        diagnosis_csv = args.icbhi_diagnosis_csv or str(_Path(args.icbhi_audio_dir).parent / "patient_diagnosis.csv")
        demographics = args.icbhi_demographics or str(_Path(args.icbhi_audio_dir).parent / "demographic_info.txt")
        index = ICBHIIndex(
            audio_dir=args.icbhi_audio_dir,
            patient_diagnosis_csv=diagnosis_csv,
            demographic_info_txt=demographics,
            target_diagnosis=args.icbhi_target_diagnosis,
        )
        dataset = ICBHIAudioTextDataset(cfg, index)
        loader = DataLoader(dataset, batch_size=cfg.stage1_batch_size, shuffle=True,
                             collate_fn=collate_icbhi_audio_text, drop_last=True)
        real_data = True
    else:
        dataset = DummyAudioTextDataset(cfg, n_samples=args.n_samples)
        loader = DataLoader(dataset, batch_size=cfg.stage1_batch_size, shuffle=True,
                             collate_fn=collate_audio_text, drop_last=True)
        real_data = False

    text_encoder = FrozenTextEncoder(cfg)  # loads Phi-2, frozen — this is the slow/heavy part

    proj = ProjectionHead.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=cfg.stage1_lr)

    proj.train()
    for epoch in range(cfg.stage1_epochs):
        epoch_loss = 0.0
        batch_iter = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.stage1_epochs}", leave=False)
        for batch in batch_iter:
            if real_data:
                _ids, waveform, texts = batch  # ids unused now — CLAP runs live, no by-id feature cache
            else:
                waveform, texts = batch
            waveform = waveform.to(device)

            with torch.no_grad():
                audio_emb = audio_encoder(waveform).float().to(device)
                text_emb = text_encoder.encode(texts).to(device)

            z_a = proj(audio_emb)
            loss = clip_contrastive_loss(z_a, text_emb, temperature=cfg.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if (epoch + 1) % max(1, cfg.stage1_epochs // 20) == 0 or epoch == 0:
            print(f"[stage1] epoch {epoch+1}/{cfg.stage1_epochs}  loss={avg:.4f}")

    save_checkpoint(proj.state_dict(), args.out)


if __name__ == "__main__":
    main()
