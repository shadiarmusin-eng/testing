"""Служебное: счётчики вызовов кернелов, одноразовые логи, печать итогов при выходе.

Нужно, чтобы по логам `soup train` было видно, какие патчи реально сработали (и не молчали ли
откаты), и чтобы dtype-реальность модели была подтверждена цифрами, а не догадками. Никаких
синхронизаций GPU в горячем пути: счётчики — просто целые числа на CPU.
"""

import atexit
import sys

COUNTS = {}   # имя -> число вызовов (наш кернел или откат)
ONCE = set()  # уже напечатанные ключи log_once


def bump(name, n=1):
    """Увеличить счётчик вызовов. Дёшево и безопасно для горячего пути."""
    COUNTS[name] = COUNTS.get(name, 0) + n


def log_once(key, msg):
    """Напечатать сообщение с префиксом [solution] ровно один раз на ключ."""
    if key in ONCE:
        return
    ONCE.add(key)
    try:
        print(f"[solution] {msg}", flush=True)
    except Exception:  # печать никогда не должна ронять обучение
        pass


def _dump():
    """Печать всех счётчиков при выходе процесса (видно в runs/<run>/train.log)."""
    try:
        if not COUNTS:
            return
        parts = ", ".join(f"{k}={v}" for k, v in sorted(COUNTS.items()))
        print(f"[solution] counters: {parts}", flush=True)
    except Exception:
        print("[solution] counters: <dump failed>", file=sys.stderr, flush=True)


atexit.register(_dump)


def is_fp16_autocast():
    """True, если сейчас идёт torch.autocast('cuda', fp16) — условие для out16-режима RMSNorm.

    Пробуем новый API torch.amp.*, при ImportError/TypeError тихо откатываемся на старый
    torch.is_autocast_enabled()/torch.get_autocast_dtype().
    """
    try:
        import torch

        try:
            enabled = torch.amp.is_autocast_available("cuda") and \
                torch.amp.is_autocast_enabled("cuda")
            dtype = torch.amp.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):  # старый torch (< 2.4-ish)
            enabled = torch.is_autocast_enabled()
            getter = getattr(torch, "get_autocast_gpu_dtype", None) or \
                getattr(torch, "get_autocast_dtype", None)
            dtype = getter() if callable(getter) else torch.float16
        return bool(enabled) and dtype == torch.float16
    except Exception:
        return False
