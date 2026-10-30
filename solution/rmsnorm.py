"""Fused RMSNorm для Qwen2: свой @triton.jit вперёд и назад (написано с нуля).

Идея: стандартная формула RMSNorm (Zhang & Sennrich, 2019) и раскладка «одна строка = один
program». Эталонный Qwen2RMSNorm — это ~7 отдельных CUDA-операций вперёд и ещё больше назад,
плюс fp32-копии активаций для backward. Здесь вся норма — один запуск вперёд и один назад.

Округления повторяют эталон: (x*rstd) приводится к dtype ВХОДА, затем умножается на вес.

Режим out16 (только внутри fp16-autocast, только для норм слоёв): выход сразу fp16. Дальше
идут bnb-слои с compute_dtype=fp16, которые всё равно привели бы вход к fp16, поэтому три
каста на слой исчезают, а fp32-копия (B,T,H) не создаётся. Финальную норму не трогаем: её
выход читает lm_head.

Веса норм заморожены (LoRA), градиент по весу не считаем; если вес вдруг требует grad — откат.
"""

import torch
import triton
import triton.language as tl

from . import config
from ._util import bump, is_fp16_autocast, log_once, note_seq, timed


@triton.jit
def soup_rmsnorm_fwd_kernel(X, W, Y, R, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=msk, other=0.0)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    r = tl.rsqrt(var + eps)
    # то же округление, что у эталона: (x * r).to(dtype входа)
    xn = (xf * r).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    tl.store(Y + row.to(tl.int64) * stride_y + cols, (xn * w).to(Y.dtype.element_ty), mask=msk)
    tl.store(R + row, r)


@triton.jit
def soup_rmsnorm_bwd_kernel(DY, X, W, R, DX, stride_dy, stride_x, stride_dx, N,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    dy = tl.load(DY + row.to(tl.int64) * stride_dy + cols, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=msk, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    r = tl.load(R + row)
    xn = x * r
    dxn = dy * w
    c = tl.sum(dxn * xn, axis=0) / N
    dx = r * (dxn - xn * c)          # dx = r*(gx - xn*mean(gx*xn)): проверено численно
    tl.store(DX + row.to(tl.int64) * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=msk)


def _cfg(N):
    block = triton.next_power_of_2(N)
    warps = 4 if block <= 2048 else 8
    return block, warps


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out16=False):
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        weight = weight.contiguous()
        out_dtype = torch.float16 if out16 else torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
        r = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        block, warps = _cfg(N)
        with timed(("rms_fwd", str(x2.dtype), str(y.dtype))):
            soup_rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, y, r, x2.stride(0), y.stride(0),
                                                    N, eps, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, r)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, r = ctx.saved_tensors
        N = x2.shape[-1]
        dy2 = dy.reshape(-1, N)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        block, warps = _cfg(N)
        with timed(("rms_bwd", str(dy2.dtype), str(x2.dtype))):
            soup_rmsnorm_bwd_kernel[(x2.shape[0],)](dy2, x2, weight, r, dx, dy2.stride(0),
                                                    x2.stride(0), dx.stride(0), N,
                                                    BLOCK=block, num_warps=warps)
        return dx.view(ctx.shape), None, None, None


_SUPPORTED = (torch.float16, torch.float32)
_MARK = "_soup_out16"


def _tag_decoder_layers(m):
    """Помечаем input_layernorm / post_attention_layernorm каждого слоя при создании слоя."""
    cls = getattr(m, "Qwen2DecoderLayer", None)
    if cls is None or getattr(cls, "_soup_tagged", False):
        return
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(self, name, None)
            if mod is not None:
                setattr(mod, _MARK, True)

    cls.__init__ = __init__
    cls._soup_tagged = True


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2 as m

    cls = m.Qwen2RMSNorm
    if getattr(cls, "_soup_fused", False):
        return
    orig = cls.forward
    _tag_decoder_layers(m)

    def forward(self, hidden_states):
        w = self.weight
        if (config.ENABLE_RMSNORM and hidden_states.is_cuda and w.is_cuda
                and not w.requires_grad and hidden_states.dtype in _SUPPORTED
                and w.dtype in _SUPPORTED and w.is_contiguous()):
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            out16 = bool(config.ENABLE_RMSNORM_FP16_OUT and getattr(self, _MARK, False)
                         and is_fp16_autocast())
            bump("rmsnorm_out16" if out16 else "rmsnorm")
            if hidden_states.dim() == 3 and not getattr(self, _MARK, False):
                note_seq(hidden_states.shape[1])      # финальная норма: один раз за forward
            log_once("rms_dtypes", f"INFO rmsnorm: x={hidden_states.dtype} w={w.dtype} "
                                   f"out16={out16}")
            return RMSNormFn.apply(hidden_states, w, float(eps), out16)
        bump("rmsnorm_fallback")
        return orig(self, hidden_states)

    cls.forward = forward
    cls._soup_fused = True
