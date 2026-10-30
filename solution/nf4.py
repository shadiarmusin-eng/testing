"""Triton-распаковка NF4 (в том числе double-quant), написано с нуля.

bitsandbytes перед каждым матмулом (вперёд и назад) распаковывает 4-битный вес в плотную fp16.
Здесь то же самое одним нашим кернелом: байт хранит два 4-битных кода (старший полубайт первым),
у каждого блока из 64 элементов свой scale (absmax), значение = code[nibble] * absmax.
При double-quant сам absmax хранится как uint8: absmax = code2[q] * absmax2[блок//256] + offset.
Эта распаковка тоже сделана внутри кернела (у bnb это 3 отдельных запуска).

ВАЖНО (баг старой версии): MatMul4Bit передаёт в dequantize_4bit ТРАНСПОНИРОВАННЫЙ упакованный
вес формы (1, n/2) и ждёт обратно out.t() (в bnb: `if A.shape[0] == 1: return out.t()`).
Старое условие `A.shape[0] != 1` отсекало именно этот случай, и кернел в обучении не вызывался.
Здесь оба случая обработаны. Тест test_nf4_patch_matches_bnb сверяет это с самим bnb.

Идея формата: QLoRA (Dettmers et al., 2023) и документация bnb. Таблица NF4 берётся из
quant_state.code (то, чем bnb квантовал), поэтому не может разойтись с версией bnb.
Любой необычный случай или исключение — откат на оригинальную функцию bnb.
"""
import sys

import torch
import triton
import triton.language as tl

from . import config
from ._util import bump, log_once, timed

_BLOCK = 1024
_CODE = {}
_BROKEN = False


@triton.jit
def soup_nf4_dequant_kernel(A, ABS, CODE, ABS2, CODE2, OFFSET, OUT, n_out,
                            BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
                            NESTED: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)          # индекс ВЫХОДНОГО элемента
    m = idx < n_out
    byte = tl.load(A + (idx >> 1), mask=m, other=0).to(tl.int32)
    nib = (byte >> ((1 - (idx & 1).to(tl.int32)) * 4)) & 15   # чётный idx -> старший полубайт
    code = tl.load(CODE + nib)
    blk = idx // BLOCKSIZE
    if NESTED:
        q = tl.load(ABS + blk, mask=m, other=0).to(tl.int32)
        c2 = tl.load(CODE2 + q)
        a2 = tl.load(ABS2 + blk // BLOCKSIZE2, mask=m, other=0.0)
        off = tl.load(OFFSET)
        scale = c2 * a2 + off
    else:
        scale = tl.load(ABS + blk, mask=m, other=0.0)
    tl.store(OUT + idx, (code * scale).to(OUT.dtype.element_ty), mask=m)


def _code(st, device):
    key = str(device)
    c = _CODE.get(key)
    if c is None:
        c = getattr(st, "code", None)
        if torch.is_tensor(c) and c.numel() == 16:
            c = c.to(device)
        else:
            import bitsandbytes.functional as BF
            c = BF.get_4bit_type("nf4", device=device)
        c = c.to(torch.float32).contiguous()
        _CODE[key] = c
    return c


def _supported(A, st):
    try:
        if not (A.is_cuda and A.dtype == torch.uint8):
            return False
        if getattr(st, "quant_type", None) != "nf4":
            return False
        if st.dtype not in (torch.float16, torch.float32) or len(st.shape) != 2:
            return False
        if A.numel() * 2 != st.shape.numel() or int(st.blocksize) % 2 != 0:
            return False
        if st.nested:
            s2 = st.state2
            return (st.absmax.dtype == torch.uint8 and torch.is_tensor(st.offset)
                    and st.offset.dtype == torch.float32 and st.offset.is_cuda
                    and s2.absmax.dtype == torch.float32 and s2.code.dtype == torch.float32
                    and s2.code.numel() == 256)
        return True
    except Exception:
        return False


def dequantize_nf4(A, st):
    """Плотная матрица st.shape (dtype st.dtype) из упакованного A. Транспонирование — снаружи."""
    n_out = st.shape.numel()
    out = torch.empty(st.shape, dtype=st.dtype, device=A.device)
    flat = A.reshape(-1)
    code = _code(st, A.device)
    grid = (triton.cdiv(n_out, _BLOCK),)
    with timed(("nf4", bool(st.nested), str(st.dtype))):
        if st.nested:
            s2 = st.state2
            soup_nf4_dequant_kernel[grid](flat, st.absmax, code, s2.absmax, s2.code, st.offset,
                                          out, n_out, BLOCKSIZE=int(st.blocksize),
                                          BLOCKSIZE2=int(s2.blocksize), NESTED=True,
                                          BLOCK=_BLOCK, num_warps=4)
        else:
            absmax = st.absmax if st.absmax.dtype == torch.float32 else st.absmax.float()
            soup_nf4_dequant_kernel[grid](flat, absmax, code, code, code, code, out, n_out,
                                          BLOCKSIZE=int(st.blocksize), BLOCKSIZE2=1,
                                          NESTED=False, BLOCK=_BLOCK, num_warps=4)
    return out


def patch_bnb_dequant():
    import bitsandbytes.functional as BF

    orig = BF.dequantize_4bit
    if getattr(orig, "_soup_fused", False):
        return

    def dequantize_4bit(A, *args, **kwargs):
        global _BROKEN
        if config.ENABLE_NF4 and not _BROKEN:
            # узнаём quant_state независимо от того, позиционно он передан или по имени
            st = kwargs.get("quant_state")
            if st is None and len(args) >= 1 and hasattr(args[0], "absmax"):
                st = args[0]
            simple = (len(args) <= 1 and kwargs.get("absmax") is None
                      and kwargs.get("out") is None
                      and set(kwargs) <= {"quant_state", "absmax", "out", "blocksize",
                                          "quant_type"})
            if st is not None and simple and _supported(A, st):
                try:
                    res = dequantize_nf4(A, st)
                    bump("nf4_dequant")
                    log_once("nf4_info", f"INFO nf4: st.dtype={st.dtype} nested={st.nested} "
                                         f"A.shape={tuple(A.shape)}")
                    return res.t() if A.shape[0] == 1 else res
                except Exception as exc:  # обучение не должно падать из-за ускорения
                    _BROKEN = True
                    log_once("nf4_broken", f"WARN: nf4 kernel disabled after error: {exc!r}")
        bump("nf4_fallback")
        return orig(A, *args, **kwargs)

    dequantize_4bit._soup_fused = True
    dequantize_4bit._soup_orig = orig
    BF.dequantize_4bit = dequantize_4bit
    for name, mod in list(sys.modules.items()):
        if name.startswith("bitsandbytes") and getattr(mod, "dequantize_4bit", None) is orig:
            mod.dequantize_4bit = dequantize_4bit
