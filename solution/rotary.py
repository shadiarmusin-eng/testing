"""Fused rotary embedding (RoPE) для Qwen2: свой @triton.jit вперёд и назад.

В transformers apply_rotary_pos_emb — это цепочка из ~6 поэлементных операций вперёд и ~8 назад
на q и k каждого слоя (cat, neg, mul, mul, add ...). Здесь вся ротация — один запуск кернела.
Backward — та же ротация с обратным знаком, поэтому активации для backward не хранятся.

Формулы (rotate_half(x) = cat(-x2, x1)):
  вперёд:  y1 = x1*c1 - x2*s1,  y2 = x2*c2 + x1*s2
  назад:   dx1 = dy1*c1 + dy2*s2,  dx2 = dy2*c2 - dy1*s1
Идея — стандартный RoPE (Su et al., 2021); код написан с нуля. Один program обрабатывает все
головы одной позиции (b, s), поэтому cos/sin грузятся один раз, а q читается подряд (раскладка
после view+transpose у q такая, что головы одной позиции лежат рядом).
"""
import torch
import triton
import triton.language as tl

from . import config
from ._util import bump, is_fp16_autocast, log_once, timed


@triton.jit(do_not_specialize=["S", "H"])
def soup_rope_kernel(X, COS, SIN, Y,
                     sx_b, sx_h, sx_s, sy_b, sy_h, sy_s, sc_b, sc_s,
                     S, H, HALF: tl.constexpr, HB: tl.constexpr, BACKWARD: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    hh = tl.arange(0, HB)
    dd = tl.arange(0, HALF)
    hm = hh < H
    xbase = X + b.to(tl.int64) * sx_b + s.to(tl.int64) * sx_s
    ybase = Y + b.to(tl.int64) * sy_b + s.to(tl.int64) * sy_s
    cbase = b.to(tl.int64) * sc_b + s.to(tl.int64) * sc_s
    c1 = tl.load(COS + cbase + dd).to(tl.float32)
    c2 = tl.load(COS + cbase + HALF + dd).to(tl.float32)
    s1 = tl.load(SIN + cbase + dd).to(tl.float32)
    s2 = tl.load(SIN + cbase + HALF + dd).to(tl.float32)
    xo = hh[:, None].to(tl.int64) * sx_h + dd[None, :]
    yo = hh[:, None].to(tl.int64) * sy_h + dd[None, :]
    x1 = tl.load(xbase + xo, mask=hm[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(xbase + xo + HALF, mask=hm[:, None], other=0.0).to(tl.float32)
    if BACKWARD:
        y1 = x1 * c1[None, :] + x2 * s2[None, :]
        y2 = x2 * c2[None, :] - x1 * s1[None, :]
    else:
        y1 = x1 * c1[None, :] - x2 * s1[None, :]
        y2 = x2 * c2[None, :] + x1 * s2[None, :]
    tl.store(ybase + yo, y1.to(Y.dtype.element_ty), mask=hm[:, None])
    tl.store(ybase + yo + HALF, y2.to(Y.dtype.element_ty), mask=hm[:, None])


def _launch(x, cos, sin, y, backward):
    B, H, S, D = x.shape
    sc_b = cos.stride(0) if cos.shape[0] > 1 else 0
    with timed(("rope", backward, str(x.dtype), str(cos.dtype))):
        soup_rope_kernel[(B * S,)](x, cos, sin, y,
                                   x.stride(0), x.stride(1), x.stride(2),
                                   y.stride(0), y.stride(1), y.stride(2),
                                   sc_b, cos.stride(1), S, H,
                                   HALF=D // 2, HB=max(16, triton.next_power_of_2(H)),
                                   BACKWARD=backward, num_warps=4)


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, out_dtype):
        y = torch.empty_like(x, dtype=out_dtype)          # сохраняет раскладку x
        _launch(x, cos, sin, y, False)
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        dx = torch.empty_like(dy, dtype=ctx.x_dtype)
        _launch(dy, cos, sin, dx, True)
        return dx, None, None, None


def _applicable(q, k, cos, sin, unsq):
    return (config.ENABLE_ROTARY and unsq == 1 and q.is_cuda and k.is_cuda
            and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3 and sin.dim() == 3
            and q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0
            and (q.shape[-1] // 2) & (q.shape[-1] // 2 - 1) == 0       # HALF — степень двойки
            and q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2]
            and q.stride(-1) == 1 and k.stride(-1) == 1
            and q.dtype in (torch.float16, torch.float32) and k.dtype == q.dtype
            and cos.shape[0] in (1, q.shape[0]) and cos.shape[1] == q.shape[2]
            and cos.shape[2] == q.shape[-1] and cos.shape == sin.shape
            and cos.stride(-1) == 1 and sin.stride() == cos.stride()
            and cos.dtype in (torch.float16, torch.float32))


def patch_qwen2_rotary():
    from transformers.models.qwen2 import modeling_qwen2 as m

    orig = m.apply_rotary_pos_emb
    if getattr(orig, "_soup_fused", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        unsq = kwargs.get("unsqueeze_dim", args[1] if len(args) >= 2 else 1)
        if _applicable(q, k, cos, sin, unsq):
            # внутри fp16-autocast SDPA всё равно привёл бы q и k к fp16 — отдаём fp16 сразу
            out_dtype = q.dtype if is_fp16_autocast() else torch.promote_types(q.dtype, cos.dtype)
            bump("rotary")
            log_once("rope_dtypes", f"INFO rotary: q={q.dtype} cos={cos.dtype} out={out_dtype} "
                                    f"q.shape={tuple(q.shape)}")
            return (_RopeFn.apply(q, cos, sin, out_dtype),
                    _RopeFn.apply(k, cos, sin, out_dtype))
        bump("rotary_fallback")
        return orig(q, k, cos, sin, *args, **kwargs)

    apply_rotary_pos_emb._soup_fused = True
    apply_rotary_pos_emb._soup_orig = orig
    m.apply_rotary_pos_emb = apply_rotary_pos_emb
