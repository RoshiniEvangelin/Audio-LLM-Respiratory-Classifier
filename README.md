# RespiraMFM — implementation scaffold

*Built to get hands-on, end to end, with the full pipeline the paper describes —
contrastive audio-language alignment followed by LoRA fine-tuning of an LLM on
top of it — including the parts that don't show up in a paper: no official code
release to check against, dataset metadata that wasn't where the docs said it'd
be, and adapting a Linux-first architecture to run cleanly on Windows.*

A working PyTorch reimplementation scaffold of **"RespiraMFM: A Multimodal
Foundation Model with Contrastive Audio-Language Alignment for Respiratory
Disease Identification"** (arXiv:[2606.09966](https://arxiv.org/abs/2606.09966),
ACL 2026). No official code release was found at the time this was built
(checked the arXiv page, HTML version, and Papers with Code) — this was
built from the paper's described architecture and hyperparameters.

Everything here **runs end-to-end today** on random dummy data, so you can
verify the pipeline is wired correctly before spending time on real data.
Two things are intentionally mocked/substituted and clearly marked for
swapping:

1. **Audio encoder** — the paper uses OPERA-CT, but that requires a
   Linux-only conda environment (shell scripts, pinned TensorFlow 2.15 +
   PyTorch 2.3, `source ~/.bashrc`) that's painful on Windows. This scaffold
   uses **CLAP** (`laion/clap-htsat-unfused`) instead — installs with plain
   `pip install transformers`, pure PyTorch, works identically on
   Windows/Mac/Linux. It's a defensible substitution on paper-fidelity
   grounds too: RespiraMFM's own BTS baseline uses CLAP for this exact role.
   See `models/audio_encoder.py` for the full rationale. `MockAudioEncoder`
   (untrained CNN stand-in) is the default until you pass `--use_clap`.
2. **Datasets** — `data.py`'s `Dummy*Dataset` classes generate random audio +
   templated symptom text by default. `data_icbhi.py` has a real loader for
   the ICBHI 2017 dataset (see below); other datasets need a similar loader
   written for their own metadata format.

## Quickstart (smoke test, no GPU required, no real data needed)

```bash
pip install -r requirements.txt

# Stage 1: contrastive audio-text alignment (tiny run)
python train_stage1.py --epochs 3 --n_samples 32 --batch_size 8

# Stage 2: LoRA fine-tuning for classification (tiny run)
python train_stage2.py --epochs 1 --n_samples 16 --batch_size 2

# Try inference
python inference.py
```

The first run downloads Phi-2 (2.7B params, ~5.5GB) from HuggingFace — that's
the slow part, not the training loop.

## Architecture, as implemented

| Component | File | Paper spec | This scaffold |
|---|---|---|---|
| Audio encoder | `models/audio_encoder.py` | OPERA-CT, frozen, 768-dim out | **CLAP, frozen, 512-dim out** |
| Projection head (Stage 1) | `models/projection.py` | 768→1024→2560 MLP, LayerNorm+ReLU+Dropout(0.1) | 512→1024→2560 (same shape, smaller input) |
| Contrastive loss | `losses.py` | InfoNCE, τ=0.07 | same |
| Text encoder (frozen) | `models/llm_backbone.py::FrozenTextEncoder` / `TextEncoder` | Phi-2, last-token pooled | same |
| Backbone + fusion | `models/llm_backbone.py::RespiraMFM` | Phi-2 + LoRA (r=16, α=32, dropout=0.1) | same |
| Classifier head | `models/llm_backbone.py::RespiraMFM.classifier` | Linear(2560 → num_classes) | same |

Hyperparameters (`configs.py`) are transcribed directly from the paper where
stated: Stage 1 — 500 epochs, lr 1e-3, τ=0.07. Stage 2 — 20 epochs, batch
size 16, max seq len 256, lr 1e-5, weight decay 0.1, LoRA r=16/α=32/dropout=0.1,
linear warmup, AdamW, 4×A100-80GB in the paper (see single-GPU notes below).

## Results (ICBHI 2017, COPD-vs-rest, held out by patient)

Stage 1 (contrastive alignment) + Stage 2 (LoRA fine-tune) trained on 734
recordings from 101 patients; evaluated on 186 recordings from 25 patients
never seen during training (patients are split before recordings, so no
patient's audio leaks across the train/test boundary):

| Metric | Held-out result | Majority-class baseline |
|---|---|---|
| Accuracy | 0.909 | 0.817 |
| F1 | 0.947 | — |
| AUROC | 0.967 | 0.500 |

The baseline column matters here: 81.7% of the held-out set is COPD-positive
(ICBHI's recording-level class balance is skewed — see caveat below), so a
model that always predicts "COPD" scores 0.817 accuracy for free. AUROC is
the metric that isn't fooled by that imbalance, and 0.967 indicates the
model is genuinely discriminating COPD from non-COPD audio, not just
exploiting the skew. Reproduce with:

```bash
python train_stage2.py --use_clap \
    --icbhi_audio_dir "path/to/ICBHI_final_database" \
    --icbhi_target_diagnosis COPD --epochs 3
```

(`--test_frac` controls the patient-level held-out fraction, default 0.2.)

**Two honest caveats**:

1. The paper's fusion mechanism (how exactly z_audio, z_prompt, z_context
   get combined and fed into the LLM) was extracted via automated PDF/HTML
   parsing to build this scaffold, and the described equations were terse
   enough to admit two readings. I implemented the more literal one (a
   3-token fused sequence) — see the design note at the top of
   `models/llm_backbone.py` for the alternative and how to swap it in.
2. Swapping OPERA-CT for CLAP means Stage 1's contrastive alignment starts
   from a different (arguably better — CLAP is already audio-text aligned)
   point than the paper, and any AUROC numbers you get won't be directly
   comparable to the paper's reported results. Fine for learning the
   architecture and building a working system; flag this explicitly if you
   ever write up results against the paper's numbers.

## Real data: ICBHI 2017 (Respiratory Sound Database)

ICBHI is the easiest real dataset to start with — freely downloadable, no
data use agreement, from [bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).
Extract the zip; you should get a folder of `.wav` + matching `.txt`
(per-breathing-cycle crackle/wheeze annotations) files, plus
`patient_diagnosis.csv` and `demographic_info.txt`.

`data_icbhi.py` reads that layout directly — see its module docstring for
the exact expected structure. It builds the "context" text Stage 2 needs
from what ICBHI actually provides (age, sex, plus a crackle/wheeze summary
per recording) rather than inventing a symptom checklist the dataset doesn't
have.

```bash
python train_stage1.py --use_clap \
    --icbhi_audio_dir "path/to/ICBHI_final_database" \
    --epochs 5   # try a small epoch count first; full paper spec is 500

python train_stage2.py --use_clap \
    --icbhi_audio_dir "path/to/ICBHI_final_database" \
    --icbhi_target_diagnosis COPD \
    --epochs 3
```

If `patient_diagnosis.csv` / `demographic_info.txt` aren't sitting one level
above the audio folder, pass `--icbhi_diagnosis_csv` / `--icbhi_demographics`
explicitly. `--icbhi_target_diagnosis` picks the one-vs-rest classification
task (default `COPD`, matching the paper's ICBHI task T4) — also supports
`Asthma` and `Pneumonia` out of the box (add more to `TASK_PROMPTS` in
`configs.py` for URTI/LRTI/Bronchiectasis/Bronchiolitis).

Other datasets from the paper, for later:

| Dataset | Disease | Access |
|---|---|---|
| ICBHI | COPD | publicly downloadable — wired up above |
| Coughvid | COVID-19 | publicly downloadable |
| Coswara | COVID-19 | publicly downloadable |
| UK COVID-19 Sounds | COVID-19 | requires request/agreement |
| TBscreen | TB | requires request |
| CodaTB | TB | requires request |
| KAUH | COPD/asthma/pneumonia | check current terms |

Each new dataset needs its own loader like `data_icbhi.py` — the main work
per dataset is the same "structured metadata → templated context text"
conversion the paper describes (Table 5), adapted to whatever that
dataset actually provides (ICBHI has no symptom checklist; UK COVID-19
Sounds does, via its app).

## Adapting to a single GPU

The paper trains on 4×A100-80GB; a single GPU (e.g. a T4/L4/3090) needs a
few adjustments, all already wired into the config:

- **LoRA is already on** — only ~0.1–1% of Phi-2's params are trainable, so
  the big memory cost is activations + the frozen base weights, not gradients.
- Set `cfg.use_4bit = True` (or `--use_4bit` on `train_stage2.py`) to load
  Phi-2 in 4-bit via bitsandbytes if you're on a ≤16GB GPU.
- Drop `stage2_batch_size` (e.g. to 2–4) and rely on the LR schedule scaling
  reasonably, or add gradient accumulation if you need the effective batch
  size closer to 16 (not yet wired in `train_stage2.py` — a ~5 line addition
  if you need it: accumulate `loss / accum_steps` over N mini-batches before
  `optimizer.step()`).
- If Phi-2 is still too heavy for iterating quickly, swap `cfg.llm_name` to
  something smaller (`"microsoft/phi-1_5"`, `"gpt2"`) for development, and
  switch back to Phi-2 for a final real run — everything else in the
  pipeline is architecture-agnostic to the LLM's hidden size EXCEPT
  `cfg.llm_hidden_dim` and `cfg.lora_target_modules`, which you'd update to
  match (e.g. GPT-2 uses `c_attn`/`c_proj` instead of `q_proj`/`k_proj`/etc.).
- CLAP itself (~600MB) is far lighter than OPERA-CT's full pipeline and runs
  live per-batch rather than needing a separate precompute step — no extra
  GPU-memory planning needed for it specifically.

## Suggested order of attack

1. Run the quickstart above as-is — confirms tensor shapes, loss goes down
   on dummy data, checkpoints save/load correctly.
2. Download ICBHI, run Stage 1 + Stage 2 with `--use_clap
   --icbhi_audio_dir ...` at a small epoch count — confirms the real-data
   path works end to end.
3. Sanity-check Stage 1 alignment (e.g. a quick t-SNE of the projected
   embeddings — same patient's audio/text should cluster, different
   patients shouldn't) before trusting Stage 2 results.
4. Scale up epochs on ICBHI alone, check AUROC on a held-out split (not just
   the loss curve) before adding more datasets.
5. Only then consider wiring up Coughvid/Coswara for the zero-shot-style
   evaluation the paper does.

## Author

Built by [Roshini Evangelin Tamanamu](https://github.com/RoshiniEvangelin) —
M.S. Computer Science candidate at Purdue University Northwest, Graduate
Research Assistant at the CIVS Lab.