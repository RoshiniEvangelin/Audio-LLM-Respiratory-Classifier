"""
Stage 1 contrastive loss.

The paper writes the one-directional (audio->text) InfoNCE form:

    L = -1/N * sum_i log( exp(z_a_i . z_t_i / tau) / sum_j exp(z_a_i . z_t_j / tau) )

`clip_contrastive_loss` below implements that, and also (default) the
symmetric CLIP-style version — averaging the audio->text and text->audio
directions — which is standard practice and more stable to train; the
one-directional paper formula is exposed via `symmetric=False` if you want
to match it exactly for a faithful reproduction.
"""

import torch
import torch.nn.functional as F


def clip_contrastive_loss(z_audio: torch.Tensor, z_text: torch.Tensor,
                           temperature: float = 0.07, symmetric: bool = True) -> torch.Tensor:
    """
    z_audio, z_text: (B, D) — will be L2-normalized internally.
    Positives are same-index pairs (z_audio[i] <-> z_text[i]).
    """
    z_audio = F.normalize(z_audio, dim=-1)
    z_text = F.normalize(z_text, dim=-1)

    logits = z_audio @ z_text.t() / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_a2t = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_a2t

    loss_t2a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a2t + loss_t2a)
