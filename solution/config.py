"""Флаги решения: что включено (True) и что нет (False).

Каждый патч проверяет свой флаг и при любой ошибке откатывается на оригинальный код, поэтому
любой флаг можно выключить, не трогая остальные файлы. Если check_rules.py или тесты
ругаются, выключайте флаги в таком порядке (см. инструкцию): RMSNORM_FP16_OUT, MLP_FAST,
NF4, CE_INPLACE_GRAD.
"""

ENABLE_RMSNORM = True            # наш Triton RMSNorm вместо 5 eager-операций transformers
ENABLE_RMSNORM_FP16_OUT = True   # нормы слоёв отдают fp16 сразу (меньше кастов и памяти)
ENABLE_ROTARY = True             # fused RoPE для q и k (rotary.py)
ENABLE_MLP_FAST = True           # MLP целиком в fp16 + fused SwiGLU (mlp.py)
ENABLE_CE = True                 # fused cross-entropy по словарю 152k (fused_ce.py)
ENABLE_ENTROPY = True            # fused entropy (TRL считает её каждый шаг)
ENABLE_NF4 = True                # Triton-распаковка NF4 (nf4.py)
CE_INPLACE_GRAD = False          # писать градиент поверх logits (-0,6 ГБ памяти).
                                 # Включать только после чистого прогона и проверки check_rules.
