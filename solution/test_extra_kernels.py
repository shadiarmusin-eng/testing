"""GPU-тесты наших кернелов против PyTorch / bitsandbytes (forward и backward).

Запуск на Kaggle: python -m pytest solution -q
Без GPU все тесты пропускаются (skip), а не падают.
"""
import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


@cuda
@pytest.mark.parametrize("xdt", [torch.float16, torch.float32])
def test_rmsnorm(xdt):
    from solution.rmsnorm import RMSNormFn
    torch.manual_seed(0)
    x = (torch.randn(3, 37, 1536, device="cuda") * 2).to(xdt).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda")
    y = RMSNormFn.apply(x, w, 1e-6)
    y_ref = ref_rmsnorm(x_ref, w, 1e-6)
    assert y.dtype == y_ref.dtype
    torch.testing.assert_close(y, y_ref, rtol=2e-3, atol=2e-3)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@cuda
def test_rmsnorm_out16():
    from solution.rmsnorm import RMSNormFn
    torch.manual_seed(0)
    x = (torch.randn(3, 37, 1536, device="cuda") * 2).requires_grad_()      # fp32 вход
    x_ref = x.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda").half()
    y = RMSNormFn.apply(x, w, 1e-6, True)
    assert y.dtype == torch.float16
    y_ref = ref_rmsnorm(x_ref, w, 1e-6).half()
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=2e-3, atol=2e-3)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@cuda
@pytest.mark.parametrize("ldt", [torch.float32, torch.float16])
@pytest.mark.parametrize("inplace", [False, True])
def test_ce_and_entropy(ldt, inplace):
    from solution.fused_ce import cross_entropy_rows, fused_entropy
    torch.manual_seed(0)
    V = 151936
    logits = (torch.randn(2 * 19, V, device="cuda") * 3).to(ldt).requires_grad_()
    ref = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, V, (2 * 19,), device="cuda")
    labels[::5] = -100
    p = torch.log_softmax(logits.detach().float(), -1)
    ent_ref = -(p.exp() * p).sum(-1)
    rows = cross_entropy_rows(logits, labels, inplace=inplace)
    ref_rows = F.cross_entropy(ref.float(), labels, ignore_index=-100, reduction="none")
    torch.testing.assert_close(rows, ref_rows, rtol=1e-4, atol=1e-4)
    g = torch.rand_like(rows)
    (rows * g).sum().backward()
    (ref_rows * g).sum().backward()
    torch.testing.assert_close(logits.grad.float(), ref.grad.float(), rtol=1e-2, atol=1e-4)
    # энтропию считаем от значений ДО backward (при inplace logits затёрты градиентом)
    ent = fused_entropy(ref.detach())
    torch.testing.assert_close(ent, ent_ref, rtol=1e-4, atol=1e-4)


def _quantize(nested, shape=(1536, 1536)):
    import bitsandbytes.functional as BF
    W = torch.randn(*shape, device="cuda", dtype=torch.float16)
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    return BF, q, st


@cuda
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_dequant(nested):
    from solution.nf4 import dequantize_nf4
    torch.manual_seed(0)
    BF, q, st = _quantize(nested)
    ref = BF.dequantize_4bit(q, st)
    out = dequantize_nf4(q, st)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@cuda
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_patch_matches_bnb(nested):
    """Патч обязан давать то же, что оригинал bnb, и для обычного A, и для транспонированного."""
    from solution import config
    from solution.nf4 import patch_bnb_dequant
    torch.manual_seed(0)
    BF, q, st = _quantize(nested, shape=(256, 1536))
    patch_bnb_dequant()                      # повторный вызов безопасен (идемпотентен)
    patched = BF.dequantize_4bit
    assert getattr(patched, "_soup_fused", False), "патч не установился"
    orig = patched._soup_orig                # настоящий оригинал bnb
    for A in (q, q.t()):              # q.t() имеет форму (1, n/2): так вызывает MatMul4Bit
        ref = orig(A, st)
        out = BF.dequantize_4bit(A, st)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@cuda
def test_fast_mlp_on_bnb_linear4bit():
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2 as m
    from solution.mlp import fast_forward

    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, hidden_act="silu")
    mlp = m.Qwen2MLP(cfg)
    for name in ("gate_proj", "up_proj", "down_proj"):
        old = getattr(mlp, name)
        new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=False,
                                compute_dtype=torch.float16, quant_type="nf4",
                                compress_statistics=True)
        new.weight = bnb.nn.Params4bit(old.weight.data, requires_grad=False,
                                       quant_type="nf4", compress_statistics=True)
        setattr(mlp, name, new.to("cuda"))
    x = torch.randn(8, 16, 64, device="cuda", requires_grad=True)          # fp32, как поток в базе
    dy = torch.randn(8, 16, 64, device="cuda")

    ref = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))         # базовый путь
    ref.backward(dy)
    ref_dx = x.grad.clone()
    x.grad = None

    out = fast_forward(mlp, x)
    assert out is not None, "быстрый путь не включился (compute_dtype != fp16?)"
    assert out.dtype == torch.float16
    out.float().backward(dy)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=5e-2, atol=5e-2)


def _rope_inputs(qdt, B=2, S=37, H=12, Hk=2, D=128):
    torch.manual_seed(0)
    qb = torch.randn(B, S, H, D, device="cuda", dtype=qdt)
    kb = torch.randn(B, S, Hk, D, device="cuda", dtype=qdt)
    freqs = torch.randn(B, S, D // 2, device="cuda")
    emb = torch.cat([freqs, freqs], dim=-1)
    return qb, kb, emb.cos(), emb.sin()


@cuda
@pytest.mark.parametrize("qdt", [torch.float16, torch.float32])
def test_rotary_matches_hf(qdt):
    import transformers.models.qwen2.modeling_qwen2 as m
    from solution.rotary import patch_qwen2_rotary
    patch_qwen2_rotary()
    ref_fn = getattr(m.apply_rotary_pos_emb, "_soup_orig", m.apply_rotary_pos_emb)
    qb, kb, cos, sin = _rope_inputs(qdt)
    q1, k1 = qb.clone().requires_grad_(), kb.clone().requires_grad_()
    q2, k2 = qb.clone().requires_grad_(), kb.clone().requires_grad_()
    qo, ko = m.apply_rotary_pos_emb(q1.transpose(1, 2), k1.transpose(1, 2), cos, sin)
    qr, kr = ref_fn(q2.transpose(1, 2), k2.transpose(1, 2), cos, sin)
    assert qo.dtype == qr.dtype and ko.dtype == kr.dtype
    torch.testing.assert_close(qo.float(), qr.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(ko.float(), kr.float(), rtol=2e-3, atol=2e-3)
    dq, dk = torch.randn_like(qr), torch.randn_like(kr)
    torch.autograd.backward([qo, ko], [dq, dk])
    torch.autograd.backward([qr, kr], [dq, dk])
    tol = 5e-3 if qdt == torch.float16 else 1e-4
    torch.testing.assert_close(q1.grad.float(), q2.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k1.grad.float(), k2.grad.float(), rtol=tol, atol=tol)


@cuda
def test_rotary_gives_fp16_inside_autocast():
    import transformers.models.qwen2.modeling_qwen2 as m
    from solution.rotary import patch_qwen2_rotary
    patch_qwen2_rotary()
    qb, kb, cos, sin = _rope_inputs(torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        qo, ko = m.apply_rotary_pos_emb(qb.transpose(1, 2), kb.transpose(1, 2), cos, sin)
    assert qo.dtype == torch.float16 and ko.dtype == torch.float16
