"""
Frozen English CTC ASR loss for UL-UNAS fine-tuning (F_GENERIC_ASR / F_ROBOT_ASR).

Uses facebook/wav2vec2-base-960h, frozen (`requires_grad=False` on every parameter),
called in `.eval()` mode -- but forward is NOT wrapped in `torch.no_grad()`, so gradients
still flow from the CTC loss back through the ASR model to the enhanced waveform and then
into UL-UNAS. This matches the brief: "frozen ASR must be in eval mode but still propagate
gradient to the enhanced waveform" (decision #22).

This is real transcript-supervised CTC loss (LibriSpeech ships transcripts), not a
representation-loss fallback.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

ASR_MODEL_NAME = "facebook/wav2vec2-base-960h"


class FrozenCTCLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(ASR_MODEL_NAME)
        self.asr = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(device)
        self.asr.eval()  # eval mode (no dropout), NOT torch.no_grad() -- gradient must flow
        for p in self.asr.parameters():
            p.requires_grad = False
        self.blank_id = self.tokenizer.pad_token_id  # 0, matches CTC blank convention here
        self.device = device

    def _normalize(self, wav):
        """Differentiable re-implementation of Wav2Vec2FeatureExtractor(do_normalize=True):
        zero-mean, unit-variance per utterance. Must be differentiable (no numpy/processor
        call) so gradient keeps flowing back to `wav`."""
        mean = wav.mean(dim=-1, keepdim=True)
        std = wav.std(dim=-1, keepdim=True)
        return (wav - mean) / (std + 1e-7)

    def encode_transcripts(self, transcripts):
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else _nullctx():
            ids = [self.tokenizer(t).input_ids for t in transcripts]
        lengths = torch.tensor([len(i) for i in ids], dtype=torch.long)
        flat = torch.tensor([tok for seq in ids for tok in seq], dtype=torch.long)
        return flat, lengths

    def forward(self, enhanced_wav, transcripts):
        """enhanced_wav: (B, n_samples) float32, requires_grad from UL-UNAS output.
        transcripts: list[str], ground-truth text (upper-case, LibriSpeech convention)."""
        x = self._normalize(enhanced_wav)
        logits = self.asr(input_values=x).logits  # (B, T, vocab), differentiable w.r.t. x
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, vocab)

        target_flat, target_lengths = self.encode_transcripts(transcripts)
        target_flat = target_flat.to(enhanced_wav.device)
        target_lengths = target_lengths.to(enhanced_wav.device)
        input_lengths = torch.full((log_probs.shape[1],), log_probs.shape[0],
                                    dtype=torch.long, device=enhanced_wav.device)

        loss = nn.functional.ctc_loss(
            log_probs, target_flat, input_lengths, target_lengths,
            blank=self.blank_id, reduction="mean", zero_infinity=True,
        )
        return loss


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gradient_scale_probe(model, se_loss_fn, asr_loss_fn, noisy, clean, transcripts, target_ratio=0.2):
    """One-shot probe: run L_SE and L_ASR independently through the same UL-UNAS forward pass,
    measure grad-norm at UL-UNAS's parameters for each, and solve lambda_asr so that
    ||lambda_asr * grad_ASR|| ~= target_ratio * ||grad_SE|| (target_ratio in [0.1, 0.3] per spec).
    Returns (lambda_asr, grad_norm_se, grad_norm_asr_raw)."""
    model.zero_grad()
    enhanced = model(noisy)
    l_se = se_loss_fn(enhanced, clean)
    l_se.backward(retain_graph=False)
    grad_norm_se = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()

    model.zero_grad()
    enhanced = model(noisy)
    l_asr = asr_loss_fn(enhanced, transcripts)
    l_asr.backward()
    grad_norm_asr_raw = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    model.zero_grad()

    if grad_norm_asr_raw < 1e-12:
        raise RuntimeError("ASR loss gradient is ~0 at UL-UNAS parameters -- gradient is not flowing through the frozen ASR model as required.")

    lambda_asr = target_ratio * grad_norm_se / grad_norm_asr_raw
    return lambda_asr, grad_norm_se, grad_norm_asr_raw
