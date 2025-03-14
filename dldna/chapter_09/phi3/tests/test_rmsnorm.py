# test_phi_mini_rmsnorm.py
import torch
import pytest
from dldna.chapter_09.phi3.simple_phi3 import PhiMiniRMSNorm  # Adjust import as needed

def test_phi_mini_rmsnorm_forward():
    hidden_size = 256
    rms_norm = PhiMiniRMSNorm(hidden_size, eps=1e-5)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = rms_norm(x)
    assert output.shape == (batch_size, seq_len, hidden_size)

def test_phi_mini_rmsnorm_eps():
    hidden_size = 256
    eps = 1e-4
    rms_norm = PhiMiniRMSNorm(hidden_size, eps=eps)
    assert rms_norm.variance_epsilon == eps

def test_phi_mini_rmsnorm_weight_initialization():
    hidden_size = 256
    rms_norm = PhiMiniRMSNorm(hidden_size)
    assert torch.allclose(rms_norm.weight, torch.ones(hidden_size))

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_phi_mini_rmsnorm_mixed_precision(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 not supported on this device")
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("CUDA not available for float16 test")
    hidden_size = 64
    rms_norm = PhiMiniRMSNorm(hidden_size, eps=1e-5).to(dtype)  # Move to device and dtype
    if dtype != torch.float32: # if fp16 or bf16, move to cuda
      rms_norm.cuda()
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda" if dtype != torch.float32 else "cpu")  # Move input to correct device
    output = rms_norm(x)
    assert output.dtype == dtype
    assert output.shape == (batch_size, seq_len, hidden_size)

def test_phi_mini_rmsnorm_backward():
    hidden_size = 128
    rms_norm = PhiMiniRMSNorm(hidden_size)
    batch_size, seq_len = 4, 8
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    output = rms_norm(x)
    loss = output.sum()
    loss.backward()
    assert rms_norm.weight.grad is not None
    assert x.grad is not None