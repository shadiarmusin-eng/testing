"""Triton cross-entropy по большому словарю + fused entropy (написано с нуля).

* cross_entropy_rows: loss по строке за один проход (online softmax); backward пишет
  (softmax - onehot) * grad за один проход. Заменяет ForCausalLMLoss из transformers
  (копия .float() + log_softmax + nll и их backward-копии размером с logits).
* fused_entropy: энтропия строки за один проход вместо entropy_from_logits из TRL
  (тот же результат, fp32-математика, без временных тензоров размером со словарь).

Идея: online softmax (Milakov & Gimelshein, 2018). Чужой код не копировался.
"""
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import config
from ._util import bump, log_once, timed

_BLOCK = 4096


@triton.jit
def soup_ce_fwd_kernel(LOGITS, LABELS, LSE, LOSS, stride, V,
                       IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)
    base = LOGITS + row.to(tl.int64) * stride
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - nm) + tl.sum(tl.exp(x - nm), axis=0)
        m = nm
    lse = m + tl.log(s)
    ignored = label == IGNORE
    safe = tl.where(ignored, 0, label)
    tgt = tl.load(base + safe).to(tl.float32)
    tl.store(LSE + row, lse)
    tl.store(LOSS + row, tl.where(ignored, 0.0, lse - tgt))


@triton.jit
def soup_ce_bwd_kernel(LOGITS, LABELS, LSE, GOUT, DX, stride, stride_dx, V,
                       IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)
    lse = tl.load(LSE + row)
    g = tl.load(GOUT + row).to(tl.float32)
    g = tl.where(label == IGNORE, 0.0, g)
    base = LOGITS + row.to(tl.int64) * stride
    dbase = DX + row.to(tl.int64) * stride_dx
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(cols == label, p - 1.0, p) * g
        tl.store(dbase + cols, p.to(DX.dtype.element_ty), mask=msk)


@triton.jit
def soup_entropy_kernel(X, OUT, stride, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = X + row.to(tl.int64) * stride
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    t = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(x, axis=0))
        a = tl.exp(m - nm)
        e = tl.exp(x - nm)
        xs = tl.where(msk, x, 0.0)
        s = s * a + tl.sum(e, axis=0)
        t = t * a + tl.sum(e * xs, axis=0)
        m = nm
    tl.store(OUT + row, m + tl.log(s) - t / s)


class _CERows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index, inplace):
        rows, V = logits.shape
        lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
        loss = torch.empty(rows, dtype=torch.float32, device=logits.device)
        with timed(("ce_fwd", str(logits.dtype))):
            soup_ce_fwd_kernel[(rows,)](logits, labels, lse, loss, logits.stride(0), V,
                                        IGNORE=ignore_index, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        ctx.inplace = bool(inplace)
        return loss

    @staticmethod
    def backward(ctx, gout):
        logits, labels, lse = ctx.saved_tensors
        rows, V = logits.shape
        gout = gout.to(torch.float32).contiguous()
        # inplace: градиент пишется поверх logits (экономит буфер размером с logits)
        dx = logits if (ctx.inplace and logits.is_contiguous()) else torch.empty_like(logits)
        with timed(("ce_bwd", str(logits.dtype))):
            soup_ce_bwd_kernel[(rows,)](logits, labels, lse, gout, dx, logits.stride(0),
                                        dx.stride(0), V, IGNORE=ctx.ignore_index,
                                        BLOCK=_BLOCK, num_warps=8)
        return dx, None, None, None


def cross_entropy_rows(logits2d, labels1d, ignore_index=-100, inplace=None):
    if inplace is None:
        inplace = config.CE_INPLACE_GRAD
    if logits2d.stride(-1) != 1:
        logits2d = logits2d.contiguous()
    return _CERows.apply(logits2d, labels1d.contiguous(), ignore_index, inplace)


def fused_entropy(logits):
    V = logits.shape[-1]
    x = logits.reshape(-1, V)
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    with timed(("entropy", str(x.dtype))):
        soup_entropy_kernel[(x.shape[0],)](x, out, x.stride(0), V, BLOCK=_BLOCK, num_warps=8)
    return out.view(logits.shape[:-1])


# ----------------------------------------------------------------- patches
def patch_loss():
    """Подменяем ForCausalLMLoss из transformers нашей версией."""
    import transformers.loss.loss_utils as lu

    orig = lu.ForCausalLMLoss
    if getattr(orig, "_soup_fused", False):
        return

    def fused_loss(logits, labels, vocab_size=None, num_items_in_batch=None,
                   ignore_index=-100, shift_labels=None, **kwargs):
        if (not config.ENABLE_CE or not logits.is_cuda or logits.dim() != 3
                or kwargs.get("reduction", None) is not None):
            bump("ce_fallback")
            return orig(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                        ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
        V = logits.shape[-1]
        if shift_labels is None:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:]
        shift_labels = shift_labels.to(logits.device).reshape(-1)
        bump("ce")
        log_once("ce_dtypes", f"INFO fused CE: logits={logits.dtype} shape={tuple(logits.shape)} "
                              f"inplace={config.CE_INPLACE_GRAD}")
        rows = cross_entropy_rows(logits.reshape(-1, V), shift_labels, ignore_index)
        total = rows.sum()
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(total.device)
            return total / num_items_in_batch
        n = (shift_labels != ignore_index).sum().clamp(min=1)
        return total / n

    fused_loss._soup_fused = True
    lu.ForCausalLMLoss = fused_loss
    mapping = getattr(lu, "LOSS_MAPPING", None)
    if isinstance(mapping, dict):
        for k, v in list(mapping.items()):
            if v is orig:
                mapping[k] = fused_loss
        mapping["ForCausalLM"] = fused_loss


def patch_entropy():
    """Подменяем entropy_from_logits из TRL везде, где она уже импортирована."""
    import trl.trainer.utils as tu

    orig = getattr(tu, "entropy_from_logits", None)
    if orig is None or getattr(orig, "_soup_fused", False):
        return

    def entropy_from_logits(logits, *args, **kwargs):
        if (config.ENABLE_ENTROPY and logits.is_cuda and logits.dim() >= 2
                and (not torch.is_grad_enabled() or not logits.requires_grad)):
            try:
                out = fused_entropy(logits)
                bump("entropy")
                return out
            except Exception as exc:  # обучение не должно падать из-за ускорения
                log_once("ent_err", f"WARN: fused entropy failed ({exc!r}); using TRL's")
        return orig(logits, *args, **kwargs)

    entropy_from_logits._soup_fused = True
    tu.entropy_from_logits = entropy_from_logits
    for name, mod in list(sys.modules.items()):
        if name.startswith("trl") and getattr(mod, "entropy_from_logits", None) is orig:
            mod.entropy_from_logits = entropy_from_logits
