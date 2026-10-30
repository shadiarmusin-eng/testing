"""Быстрый путь Qwen2MLP: один fp16-каст входа, три bnb-матмула в fp16, fused SwiGLU.

Откуда выигрыш: в базе остаточный поток fp32, а bnb Linear4bit приводит вход к fp16 и результат
обратно к fp32. Для MLP (2048 x 8960) это ~6 лишних проходов по 36-73 МБ на слой, а backward
держит fp32-копии gate/up (~220 МБ на слой). Здесь x приводится к fp16 один раз, gate/up/down
возвращают fp16, SwiGLU (пример из swiglu.py) хранит gate и up в fp16 — вдвое меньше памяти.
Числа те же: матмулы и так шли в fp16, меняется только где происходят касты.

Путь включается, только если gate/up/down — bitsandbytes Linear4bit без LoRA с
compute_dtype == float16 (compute_dtype — атрибут СЛОЯ, а не quant_state); иначе — прежний forward.
Новых @triton.jit здесь нет: используются кернелы примера, они не считаются «своими».
"""
import torch

from . import config
from ._util import bump, log_once

try:
    from bitsandbytes.nn import Linear4bit as _L4
except Exception:  # bnb нет — быстрый путь просто не включится
    _L4 = None


def _plain_bnb_fp16(mod):
    if _L4 is None or not isinstance(mod, _L4):
        return False
    if hasattr(mod, "base_layer") or hasattr(mod, "lora_A"):
        return False
    return getattr(mod, "compute_dtype", None) == torch.float16


def fast_forward(self, x):
    """Результат быстрого пути или None, если он не применим."""
    if not (config.ENABLE_MLP_FAST and x.is_cuda and x.dtype in (torch.float16, torch.float32)):
        return None
    act = getattr(getattr(self, "config", None), "hidden_act", "silu")
    if act != "silu":
        return None
    ok = getattr(self, "_soup_fast", None)
    if ok is None:
        ok = all(_plain_bnb_fp16(getattr(self, n, None))
                 for n in ("gate_proj", "up_proj", "down_proj"))
        self._soup_fast = ok
    if not ok:
        return None
    from .swiglu import swiglu
    x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    gate = self.gate_proj(x16)
    up = self.up_proj(x16)
    bump("mlp_fast")
    log_once("mlp_dtypes", f"INFO fast MLP: x={x.dtype} gate={gate.dtype}")
    return self.down_proj(swiglu(gate, up))


def patch_qwen2_mlp_fast():
    from transformers.models.qwen2 import modeling_qwen2 as m

    cls = m.Qwen2MLP
    if getattr(cls, "_soup_fast_mlp", False):
        return
    prev = cls.forward  # тот, что поставил пример swiglu (или оригинал)

    def forward(self, x):
        out = fast_forward(self, x)
        if out is None:
            bump("mlp_fallback")
            return prev(self, x)
        return out

    cls.forward = forward
    cls._soup_fast_mlp = True
