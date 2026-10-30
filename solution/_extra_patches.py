"""Точка входа всех наших патчей: apply_extra() вызывается в конце solution.apply().

Каждый патч в своём try/except и печатает «[solution] OK / SKIPPED ...», чтобы в логе было
видно, что именно включилось. Счётчики вызовов кернелов печатаются при выходе процесса
(см. _util.py): если счётчик 0 — патч молчал, и своих кернелов в обучении нет.
"""
from . import config
from ._util import log_once

_DONE = False


def apply_extra():
    global _DONE
    if _DONE:
        return
    try:
        import torch
        if not torch.cuda.is_available():  # тесты Soup могут идти без GPU
            return
    except Exception as exc:  # noqa: BLE001
        log_once("no_torch", f"SKIPPED: torch unavailable ({exc!r})")
        return

    def run(name, enabled, importer):
        if not enabled:
            log_once(f"off_{name}", f"SKIPPED {name}: выключен в config.py")
            return
        try:
            importer()()
            log_once(f"on_{name}", f"OK: {name}")
        except Exception as exc:  # noqa: BLE001
            log_once(f"err_{name}", f"SKIPPED {name}: {exc!r}")

    def rms():
        from .rmsnorm import patch_qwen2_rmsnorm
        return patch_qwen2_rmsnorm

    def rot():
        from .rotary import patch_qwen2_rotary
        return patch_qwen2_rotary

    def mlp():
        from .mlp import patch_qwen2_mlp_fast
        return patch_qwen2_mlp_fast

    def ce():
        from .fused_ce import patch_loss
        return patch_loss

    def ent():
        from .fused_ce import patch_entropy
        return patch_entropy

    def nf4():
        from .nf4 import patch_bnb_dequant
        return patch_bnb_dequant

    run("rmsnorm", config.ENABLE_RMSNORM, rms)
    run("rotary", config.ENABLE_ROTARY, rot)
    run("fast_mlp", config.ENABLE_MLP_FAST, mlp)
    run("ce_loss", config.ENABLE_CE, ce)
    run("entropy", config.ENABLE_ENTROPY, ent)
    run("nf4", config.ENABLE_NF4, nf4)
    _DONE = True
