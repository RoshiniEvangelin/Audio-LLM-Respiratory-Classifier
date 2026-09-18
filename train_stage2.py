"""
Stage 2: LoRA instruction tuning for classification.

Loads the Stage-1 projection head (frozen), fuses [z_audio; z_prompt;
z_context] into a 3-token sequence, feeds it through Phi-2 (LoRA-adapted),
and trains the LoRA weights + a linear classification head with
cross-entropy.

Run:
    python train_stage2.py --epochs 1 --n_samples 32 --batch_size 2   # smoke test, mock audio encoder
    python train_stage2.py --use_clap                                  # real CLAP features
    python train_stage2.py                                             # paper defaults (mock audio)
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from configs import Config
from data import DummyClassificationDataset, collate_classification
from models.audio_encoder import build_audio_encoder
from models.llm_backbone import RespiraMFM
from utils import set_seed, save_checkpoint


def evaluate_holdout(model, audio_encoder, loader, device, real_data):
    """Forward-only pass over a held-out split; prints accuracy/F1/AUROC.

    Unlike the per-epoch training accuracy above (which is on data the model
    is actively learning from), this is on patients never seen during this
    training run — the number that's actually meaningful to report.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    model.lm.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if real_data:
                _ids, waveform, prompts, contexts, labels = batch
            else:
                waveform, prompts, contexts, labels = batch
            waveform = waveform.to(device)
            audio_emb = audio_encoder(waveform).float()
            logits = model(audio_emb, prompts, contexts)
            probs = F.softmax(logits.float(), dim=-1)[:, 1]  # P(positive)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
    model.lm.train()

    if not all_labels:
        print("[stage2][held-out] empty held-out set — nothing to evaluate")
        return None

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    pos_rate = sum(all_labels) / len(all_labels)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auroc_str = f"{auroc:.3f}"
    except ValueError:
        auroc = float("nan")
        auroc_str = "n/a (only one class present in held-out set — try a larger --test_frac)"

    print(f"[stage2][held-out] n={len(all_labels)}  positive_rate={pos_rate:.3f}  "
          f"acc={acc:.3f}  f1={f1:.3f}  auroc={auroc_str}")
    print(f"[stage2][held-out] NOTE: a model that always predicts the majority class "
          f"would score acc={max(pos_rate, 1 - pos_rate):.3f} here — compare against that, not against 1.0.")
    return {"acc": acc, "f1": f1, "auroc": auroc, "n": len(all_labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--task", type=str, default="covid19")
    ap.add_argument("--use_clap", action="store_true",
                     help="use real pretrained CLAP audio features instead of MockAudioEncoder")
    ap.add_argument("--stage1_ckpt", type=str, default="./checkpoints/stage1_projection.pt")
    ap.add_argument("--use_4bit", action="store_true", help="load Phi-2 in 4-bit (needs bitsandbytes)")
    ap.add_argument("--out", type=str, default="./checkpoints/stage2_model.pt")
    # Real-data flags: pass --icbhi_audio_dir to train on ICBHI instead of dummy data.
    ap.add_argument("--icbhi_audio_dir", type=str, default=None)
    ap.add_argument("--icbhi_diagnosis_csv", type=str, default=None)
    ap.add_argument("--icbhi_demographics", type=str, default=None)
    ap.add_argument("--icbhi_target_diagnosis", type=str, default="COPD")
    ap.add_argument("--test_frac", type=float, default=0.2,
                     help="fraction of PATIENTS (not recordings) held out for evaluation (ICBHI only)")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.stage2_epochs = args.epochs
    if args.batch_size is not None:
        cfg.stage2_batch_size = args.batch_size

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[stage2] device={device}")

    audio_encoder = build_audio_encoder(cfg, use_clap=args.use_clap)

    if args.icbhi_audio_dir:
        from pathlib import Path as _Path
        from data_icbhi import ICBHIIndex, ICBHIClassificationDataset, collate_icbhi_classification
        diagnosis_csv = args.icbhi_diagnosis_csv or str(_Path(args.icbhi_audio_dir).parent / "patient_diagnosis.csv")
        demographics = args.icbhi_demographics or str(_Path(args.icbhi_audio_dir).parent / "demographic_info.txt")
        index = ICBHIIndex(
            audio_dir=args.icbhi_audio_dir,
            patient_diagnosis_csv=diagnosis_csv,
            demographic_info_txt=demographics,
            target_diagnosis=args.icbhi_target_diagnosis,
        )
        train_index, test_index = index.split_by_patient(test_frac=args.test_frac, seed=cfg.seed)
        train_dataset = ICBHIClassificationDataset(cfg, train_index)
        test_dataset = ICBHIClassificationDataset(cfg, test_index)
        loader = DataLoader(train_dataset, batch_size=cfg.stage2_batch_size, shuffle=True,
                             collate_fn=collate_icbhi_classification, drop_last=True)
        eval_loader = DataLoader(test_dataset, batch_size=cfg.stage2_batch_size, shuffle=False,
                                  collate_fn=collate_icbhi_classification, drop_last=False)
        real_data = True
    else:
        dataset = DummyClassificationDataset(cfg, task=args.task, n_samples=args.n_samples)
        loader = DataLoader(dataset, batch_size=cfg.stage2_batch_size, shuffle=True,
                             collate_fn=collate_classification, drop_last=True)
        eval_loader = None
        real_data = False

    model = RespiraMFM(cfg, load_in_4bit=args.use_4bit)
    try:
        model.load_stage1_projection(args.stage1_ckpt, freeze=True)
    except FileNotFoundError:
        print(f"[stage2] WARNING: no Stage-1 checkpoint at {args.stage1_ckpt} — "
              f"using a randomly-initialized (untrained) projection head. "
              f"Run train_stage1.py first for a real run.")

    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[stage2] trainable params (LoRA + classifier): {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.stage2_epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    model.lm.train()
    for epoch in range(cfg.stage2_epochs):
        epoch_loss, correct, seen = 0.0, 0, 0
        batch_iter = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.stage2_epochs}", leave=False)
        for batch in batch_iter:
            if real_data:
                _ids, waveform, prompts, contexts, labels = batch  # ids unused — CLAP runs live
            else:
                waveform, prompts, contexts, labels = batch
            waveform = waveform.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                audio_emb = audio_encoder(waveform).float()

            logits = model(audio_emb, prompts, contexts)
            loss = F.cross_entropy(logits.float(), labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += labels.size(0)

        print(f"[stage2] epoch {epoch+1}/{cfg.stage2_epochs}  "
              f"loss={epoch_loss/len(loader):.4f}  acc={correct/seen:.3f}")

    if eval_loader is not None:
        evaluate_holdout(model, audio_encoder, eval_loader, device, real_data)

    save_checkpoint(
        {"lora": {k: v for k, v in model.lm.state_dict().items() if "lora" in k},
         "classifier": model.classifier.state_dict()},
        args.out,
    )


if __name__ == "__main__":
    main()