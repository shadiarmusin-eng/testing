"""Служебное: счётчики вызовов, одноразовые логи, замер компиляции, гистограмма длин.

Печатается при выходе процесса (видно в логе soup train):
  counters   — сколько раз сработал каждый наш кернел (0 = патч молчал);
  first-call — секунды первого запуска каждого варианта кернела (≈ время компиляции Triton,
               оно входит в замер скорости);
  seq_len    — длины последовательностей на входе финальной нормы (фиксированный padding или
               динамический, сколько в нём пустых токенов).
Никаких синхронизаций GPU в горячем пути.
"""
import atexit
import sys
import time

COUNTS = {}
ONCE = set()
COMPILE = {}   # вариант кернела -> секунды первого запуска
SEQ = {}       # длина последовательности -> сколько раз видели
_SEEN = set()


def bump(name, n=1):
    COUNTS[name] = COUNTS.get(name, 0) + n


def log_once(key, msg):
    if key in ONCE:
        return
    ONCE.add(key)
    try:
        print(f"[solution] {msg}", flush=True)
    except Exception:  # печать никогда не должна ронять обучение
        pass


class timed:
    """with timed(ключ): <запуск кернела> — замеряет только ПЕРВЫЙ запуск варианта (компиляция)."""
    __slots__ = ("key", "t0")

    def __init__(self, key):
        self.key = key
        self.t0 = None if key in _SEEN else time.perf_counter()
        _SEEN.add(key)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.t0 is not None:
            COMPILE[self.key] = time.perf_counter() - self.t0
        return False


def note_seq(s):
    SEQ[int(s)] = SEQ.get(int(s), 0) + 1


def _dump():
    try:
        if COUNTS:
            print("[solution] counters: " + ", ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())),
                  flush=True)
        if COMPILE:
            total = sum(COMPILE.values())
            parts = ", ".join(f"{'/'.join(str(p) for p in k)}={v:.2f}"
                              for k, v in sorted(COMPILE.items(), key=lambda kv: -kv[1]))
            print(f"[solution] first-call seconds (compile): total={total:.1f}; {parts}", flush=True)
        if SEQ:
            ks = sorted(SEQ)
            top = sorted(SEQ.items(), key=lambda kv: -kv[1])[:6]
            print(f"[solution] seq_len: distinct={len(ks)} min={ks[0]} max={ks[-1]} "
                  f"most_common={top}", flush=True)
    except Exception:
        print("[solution] dump failed", file=sys.stderr, flush=True)


atexit.register(_dump)


def is_fp16_autocast():
    """True, если сейчас идёт torch.autocast('cuda', fp16)."""
    try:
        import torch

        try:
            enabled = torch.amp.is_autocast_available("cuda") and \
                torch.amp.is_autocast_enabled("cuda")
            dtype = torch.amp.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):
            enabled = torch.is_autocast_enabled()
            getter = getattr(torch, "get_autocast_gpu_dtype", None) or \
                getattr(torch, "get_autocast_dtype", None)
            dtype = getter() if callable(getter) else torch.float16
        return bool(enabled) and dtype == torch.float16
    except Exception:
        return False
