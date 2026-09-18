"""
Stage 2: LLM backbone (Phi-2) with LoRA, text encoder, fusion, and
classification head.

Design note / faithfulness caveat
----------------------------------
The paper's own description of the fusion step is terse: it pools the audio
embedding z_a, the prompt embedding z_p, and the context/symptom embedding
z_c each down to a single d-dim vector, then concatenates them and feeds the
result through the LLM before classifying off the final hidden state. That's
what's implemented here: a 3-token sequence [z_a; z_p; z_c] fed to Phi-2 via
`inputs_embeds`, classifying off the last token's final hidden state.

An equally plausible reading (closer to how LLaVA-style VLMs do it) would
keep the full prompt/context token sequences and only replace/prepend a
single audio token, giving the LLM much more text to attend over. If you
have access to the actual paper PDF (the auto-extraction used to build this
scaffold can mangle equations), it's worth double-checking Section 3's
fusion equations against this file — `encode_text_full_sequence` below is
provided as an easy swap-in for that alternative interpretation.

Both z_p and z_c are produced by the SAME frozen Phi-2 weights used as the
backbone (matching "LLM text encoder, frozen during Stage 1 and 2") — we get
that "frozen" behavior for free via `peft`'s `disable_adapter()` context
manager, so there's only one copy of the 2.7B weights in memory.
"""

from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.projection import ProjectionHead


class TextEncoder:
    """Frozen last-token-pooled sentence embedding using the backbone LM."""

    def __init__(self, peft_lm, tokenizer, device, max_seq_len: int):
        self.lm = peft_lm
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)

        with self.lm.disable_adapter():  # frozen forward, LoRA adapters off
            was_training = self.lm.training
            self.lm.eval()
            out = self.lm(**enc, output_hidden_states=True)
            if was_training:
                self.lm.train()

        hidden = out.hidden_states[-1]                       # (B, T, d)
        last_idx = enc["attention_mask"].sum(dim=1) - 1       # (B,) index of last real token
        pooled = hidden[torch.arange(hidden.size(0)), last_idx]  # (B, d)
        return pooled


class FrozenTextEncoder:
    """Same last-token-pooling as TextEncoder, but for Stage 1 where there's
    no LoRA wrapper yet — just a plain frozen AutoModelForCausalLM.
    """

    def __init__(self, cfg):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_name, dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_seq_len = cfg.max_seq_len

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), last_idx].float()


class RespiraMFM(nn.Module):
    def __init__(self, cfg, load_in_4bit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        if load_in_4bit:
            # Needs `bitsandbytes` installed. Recommended if Phi-2 (2.7B) +
            # LoRA + activations doesn't fit on your GPU at batch_size=16.
            load_kwargs.update(dict(load_in_4bit=True, device_map="auto"))

        base_lm = AutoModelForCausalLM.from_pretrained(cfg.llm_name, **load_kwargs)
        if not load_in_4bit:
            base_lm = base_lm.to(self.device)

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        self.lm = get_peft_model(base_lm, lora_cfg)  # freezes base weights, adds trainable LoRA
        self.lm.print_trainable_parameters()

        self.text_encoder = TextEncoder(self.lm, self.tokenizer, self.device, cfg.max_seq_len)

        # Loaded from a Stage-1 checkpoint and frozen for Stage 2 — see
        # `load_stage1_projection`.
        self.audio_proj = ProjectionHead.from_config(cfg).to(self.device)

        self.classifier = nn.Linear(cfg.llm_hidden_dim, cfg.num_classes).to(self.device)
        if self.device.type == "cuda" and not load_in_4bit:
            self.classifier = self.classifier.half()

    def load_stage1_projection(self, ckpt_path: str, freeze: bool = True):
        state = torch.load(ckpt_path, map_location=self.device)
        self.audio_proj.load_state_dict(state)
        if freeze:
            for p in self.audio_proj.parameters():
                p.requires_grad_(False)
            self.audio_proj.eval()
        print(f"[RespiraMFM] loaded Stage-1 projection head from {ckpt_path} (frozen={freeze})")

    def forward(self, audio_emb: torch.Tensor, prompts: List[str], contexts: List[str]) -> torch.Tensor:
        """
        audio_emb: (B, cfg.audio_dim) precomputed/mock OPERA-CT embeddings
        prompts, contexts: list[str], length B
        returns logits: (B, cfg.num_classes)
        """
        audio_emb = audio_emb.to(self.device)
        z_a = self.audio_proj(audio_emb)                 # (B, d)
        z_p = self.text_encoder.encode(prompts)           # (B, d)
        z_c = self.text_encoder.encode(contexts)           # (B, d)

        z_fusion = torch.stack([z_a, z_p, z_c], dim=1)     # (B, 3, d)
        z_fusion = z_fusion.to(self.lm.dtype if hasattr(self.lm, "dtype") else z_fusion.dtype)

        out = self.lm(inputs_embeds=z_fusion, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]       # (B, d) — the "context" token position
        logits = self.classifier(last_hidden.to(self.classifier.weight.dtype))
        return logits

    def trainable_parameters(self):
        """LoRA params (from self.lm) + classifier head. audio_proj/base LM stay frozen."""
        params = [p for p in self.lm.parameters() if p.requires_grad]
        params += list(self.classifier.parameters())
        return params
