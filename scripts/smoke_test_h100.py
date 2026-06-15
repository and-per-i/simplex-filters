#!/usr/bin/env python3
"""
Smoke test per H100 — Forward Triton + Backward PyTorch (w1=64, w2=256).
Verifica che:
  1. Il kernel Triton forward compili e giri
  2. Il backward PyTorch dia gradienti finiti
  3. 50 step di training non craslino
"""

import os
import sys
import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simplicial.ops.pytorch.two_simplicial_attention import (
    SimplicialAttentionFunction,
    _USE_TRITON,
    _TRITON_AVAILABLE,
)


def test_forward_backward(device, B=2, S=128, H=32, D=128, w1=64, w2=256):
    """Test forward + backward con gradienti finiti."""
    print(f"\n{'='*60}")
    print(f"Test: B={B}, S={S}, H={H}, D={D}, w1={w1}, w2={w2}")
    print(f"Device: {device} | Triton avail: {_TRITON_AVAILABLE} | USE_TRITON: {_USE_TRITON}")
    print(f"CUDA CC: {torch.cuda.get_device_capability(device) if device.type == 'cuda' else 'N/A'}")
    print(f"{'='*60}")

    q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k1 = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k2 = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v1 = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v2 = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Forward
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    output = SimplicialAttentionFunction.apply(q, k1, k2, v1, v2, w1, w2)
    torch.cuda.synchronize() if device.type == "cuda" else None
    fwd_time = time.time() - t0
    print(f"  Forward: {fwd_time*1000:.1f}ms, output shape={output.shape}, dtype={output.dtype}")

    # Verifica output finito
    assert torch.isfinite(output).all(), "❌ Forward: output non finito!"
    print(f"  ✓ Output finito: min={output.min().item():.3f}, max={output.max().item():.3f}")

    # Backward
    grad = torch.randn_like(output)
    t0 = time.time()
    output.backward(grad)
    torch.cuda.synchronize() if device.type == "cuda" else None
    bwd_time = time.time() - t0
    print(f"  Backward: {bwd_time*1000:.1f}ms")

    # Verifica gradienti finiti
    for name, tensor in [("q", q), ("k1", k1), ("k2", k2), ("v1", v1), ("v2", v2)]:
        if tensor.grad is None:
            print(f"  ❌ {name}.grad = None!")
            return False
        if not torch.isfinite(tensor.grad).all():
            print(f"  ❌ {name}.grad non finito!")
            return False
        print(f"  ✓ {name}.grad: min={tensor.grad.min().item():.3e}, max={tensor.grad.max().item():.3e}")

    print(f"\n  ✅ Test superato!")
    return True


def test_training_smoke(device, num_steps=50, S=512, H=32, D=128, w1=64, w2=256):
    """50 step di training minimale: 1 layer lineare + SimplicialAttention."""
    print(f"\n{'='*60}")
    print(f"Smoke training: {num_steps} steps, S={S}, H={H}, D={D}, w1={w1}, w2={w2}")
    print(f"{'='*60}")

    # Crea un modello minimale
    class MiniSimplicialModel(nn.Module):
        def __init__(self, H, D, w1, w2):
            super().__init__()
            self.proj_q = nn.Linear(D, H * D, bias=False)
            self.proj_k1 = nn.Linear(D, H * D, bias=False)
            self.proj_k2 = nn.Linear(D, H * D, bias=False)
            self.proj_v1 = nn.Linear(D, H * D, bias=False)
            self.proj_v2 = nn.Linear(D, H * D, bias=False)
            self.out_proj = nn.Linear(H * D, D, bias=False)
            self.w1 = w1
            self.w2 = w2
            self.H = H
            self.D = D

        def forward(self, x):
            B, S, _ = x.shape
            q = self.proj_q(x).view(B, S, self.H, self.D)
            k1 = self.proj_k1(x).view(B, S, self.H, self.D)
            k2 = self.proj_k2(x).view(B, S, self.H, self.D)
            v1 = self.proj_v1(x).view(B, S, self.H, self.D)
            v2 = self.proj_v2(x).view(B, S, self.H, self.D)

            attn = SimplicialAttentionFunction.apply(q, k1, k2, v1, v2, self.w1, self.w2)
            return self.out_proj(attn.reshape(B, S, -1))

    model = MiniSimplicialModel(H, D, w1, w2).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(num_steps):
        x = torch.randn(1, S, D, device=device, dtype=torch.bfloat16)
        target = torch.randn(1, S, D, device=device, dtype=torch.bfloat16)

        out = model(x)
        loss = F.mse_loss(out, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

    print(f"\n  ✅ {num_steps} step completati senza crash!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test H100 per simplicial attention Triton")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda|cpu)")
    parser.add_argument("--steps", type=int, default=50, help="Numero step training")
    parser.add_argument("--seq-len", type=int, default=512, help="Lunghezza sequenza")
    parser.add_argument("--w1", type=int, default=64, help="Window K1")
    parser.add_argument("--w2", type=int, default=256, help="Window K2")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  CUDA non disponibile — test solo forward+backward (senza Triton)")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        cc = torch.cuda.get_device_capability(device)
        print(f"GPU: {torch.cuda.get_device_name(device)} | CC: {cc[0]}.{cc[1]}")
        if cc[0] >= 9 and _TRITON_AVAILABLE:
            print("✅ Triton disponibile — forward su H100")
        elif cc[0] >= 9 and not _TRITON_AVAILABLE:
            print("⚠️  CC>=9 ma Triton non installato — forward PyTorch")
        else:
            print("ℹ️  CC<9 — forward PyTorch")

    # Test 1: forward + backward con gradienti finiti
    ok = test_forward_backward(
        device, B=2, S=args.seq_len, H=32, D=128, w1=args.w1, w2=args.w2
    )
    if not ok:
        sys.exit(1)

    # Test 2: 50 step di training
    test_training_smoke(
        device, num_steps=args.steps, S=args.seq_len, H=32, D=128, w1=args.w1, w2=args.w2
    )

    print(f"\n{'='*60}")
    print("Tutti i test superati! ✅")
    print(f"w1={args.w1}, w2={args.w2} H100 Triton forward + PyTorch backward ok")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()