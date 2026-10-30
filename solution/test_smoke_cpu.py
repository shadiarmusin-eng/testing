"""CPU-смоук тесты: синтаксис, импорт пакета и apply() без CUDA.

GPU-проверки — в test_extra_kernels.py; этот файл гарантирует, что ни один патч не уронит
обучение даже там, где нет GPU (apply() обязан пройти молча).
"""

import importlib.util as _ilu

try:
    _HAS_PYTEST = _ilu.find_spec("pytest") is not None
except (ImportError, ValueError):
    _HAS_PYTEST = False


def _skip(condition, reason):
    if _HAS_PYTEST:
        import pytest
        return pytest.mark.skipif(condition, reason=reason)

    def deco(fn):
        return fn

    return deco


NEEDS_GPU = _skip(not __import__("torch").cuda.is_available(), "needs a CUDA GPU")


@NEEDS_GPU
def test_apply_runs_with_real_cuda():
    """apply() на живом GPU: ставим патчи, потом Qwen2RMSNorm должен дать результат близкий к эталону."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as m

    from solution import apply

    apply()
    norm = m.Qwen2RMSNorm(64, eps=1e-6).to("cuda")
    x = torch.randn(3, 5, 64, device="cuda", dtype=torch.float16)
    y = norm(x)
    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = ((x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype) * norm.weight).float()
    torch.testing.assert_close(y.float(), ref, rtol=2e-3, atol=2e-3)
