# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Two Simplicial attention variant reference implementations.
# Note these are unoptimized references that use O(s^3) and
# WILL OOM for large seq length.

import math

import torch
import torch.nn.functional as F

# Tentativo di importare i kernel Triton (solo su H100)
try:
    from simplicial.ops.triton.bwd import triton_bwd
    from simplicial.ops.triton.fwd import triton_fwd
    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    _TRITON_AVAILABLE = False

# Triton TLX funziona solo su GPU Hopper (H100). Non ha architetture CUDA
# per compute capability < 9.0 (es. RTX 4090, A100). Su altre GPU si usa PyTorch.
_USE_TRITON = (
    _TRITON_AVAILABLE
    and torch.cuda.is_available()
    and "H100" in torch.cuda.get_device_name(0)
)


class SimplicialAttention(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        w1: int = 512,
        w2: int = 32,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.w1 = w1
        self.w2 = w2

    def forward(
        self,
        xq: torch.Tensor,
        xk1: torch.Tensor,
        xk2: torch.Tensor,
        xv1: torch.Tensor,
        xv2: torch.Tensor,
    ):
        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_heads, self.head_dim)
        xk1 = xk1.view(bs, slen, self.n_heads, self.head_dim)
        xk2 = xk2.view(bs, slen, self.n_heads, self.head_dim)
        xv1 = xv1.view(bs, slen, self.n_heads, self.head_dim)
        xv2 = xv2.view(bs, slen, self.n_heads, self.head_dim)

        output = SimplicialAttentionFunction.apply(
            xq,
            xk1,
            xk2,
            xv1,
            xv2,
            self.w1,
            self.w2,
        )
        output = output.reshape(bs, slen, -1)
        return output


def _torch_2simplicial_fwd(xq, xk1, xk2, xv1, xv2, w1, w2):
    """
    Pure PyTorch forward di attenzione 2-simpliciale — memory efficient.

    Invece di materializzare il tensore S×S×S (8.6 GB a S=512),
    itera per ogni posizione query e calcola solo la finestra locale.

    Complessità: O(B * H * S * w1 * w2 * D) invece di O(B * H * S³ * D)
    Memoria:    O(B * H * S * w1 * w2) invece di O(B * H * S³)
    """
    B, S, H, D = xq.shape
    sm_scale = 1.0 / math.sqrt(D)
    output = torch.zeros(B, S, H, D, device=xq.device, dtype=xq.dtype)

    for i in range(S):
        # Sliding window: chiavi visibili dalla posizione i
        j_start = max(0, i - w1) if w1 > 0 else 0
        j_end = i + 1 if w1 > 0 else S
        k_start = max(0, i - w2) if w2 > 0 else 0
        k_end = i + 1 if w2 > 0 else S

        # Estrai viste della finestra per questa query
        k1_window = xk1[:, j_start:j_end, :, :]  # [B, w1, H, D]
        k2_window = xk2[:, k_start:k_end, :, :]  # [B, w2, H, D]
        v1_window = xv1[:, j_start:j_end, :, :]  # [B, w1, H, D]
        v2_window = xv2[:, k_start:k_end, :, :]  # [B, w2, H, D]

        # q_i: [B, 1, H, D] → squeeze → [B, H, D]
        q_i = xq[:, i:i+1, :, :]  # [B, 1, H, D]

        # logits: [B, H, 1, w1, w2]
        logits = torch.einsum("bthd,bshd,brhd->bh tsr", q_i, k1_window, k2_window) * sm_scale
        logits = logits.squeeze(2)  # [B, H, w1, w2]

        # Softmax su [w1, w2]
        shape = logits.shape  # [B, H, w1, w2]
        logits_flat = logits.reshape(B, H, -1)
        attn = F.softmax(logits_flat, dim=-1).reshape(shape)

        # output_i: [B, H, D] = einsum('bhjk,bjhd,bkhd->bhd', attn, v1, v2)
        output_i = torch.einsum("bhjk,bjhd,bkhd->bhd", attn, v1_window, v2_window)
        output[:, i, :, :] = output_i

    return output, None


class SimplicialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xq: torch.Tensor,
        xk1: torch.Tensor,
        xk2: torch.Tensor,
        xv1: torch.Tensor,
        xv2: torch.Tensor,
        w1: int,
        w2: int,
    ):
        if _USE_TRITON:
            try:
                output, max_plus_lse = triton_fwd(xq, xk1, xk2, xv1, xv2, w1, w2)
                ctx.use_triton = True
                ctx.w1 = w1
                ctx.w2 = w2
                ctx.save_for_backward(xq, xk1, xk2, xv1, xv2, output, max_plus_lse)
                return output
            except Exception:
                pass  # fallback a PyTorch

        # Fallback PyTorch memory-efficient
        output, _ = _torch_2simplicial_fwd(xq, xk1, xk2, xv1, xv2, w1, w2)
        ctx.use_triton = False
        ctx.save_for_backward(xq, xk1, xk2, xv1, xv2)
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.sm_scale = 1.0 / math.sqrt(xq.shape[-1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_triton:
            q, k1, k2, v1, v2, output, max_plus_lse = ctx.saved_tensors
            dq, dk1, dk2, dv1, dv2 = triton_bwd(
                q, k1, k2, v1, v2, ctx.w1, ctx.w2, output, grad_output, max_plus_lse,
            )
            return dq, dk1, dk2, dv1, dv2, None, None

        # Fallback PyTorch backward
        q, k1, k2, v1, v2 = ctx.saved_tensors
        B, S, H, D = q.shape
        sm_scale = ctx.sm_scale
        w1, w2 = ctx.w1, ctx.w2

        dq = torch.zeros_like(q)
        dk1 = torch.zeros_like(k1)
        dk2 = torch.zeros_like(k2)
        dv1 = torch.zeros_like(v1)
        dv2 = torch.zeros_like(v2)

        for i in range(S):
            j_start = max(0, i - w1) if w1 > 0 else 0
            j_end = i + 1 if w1 > 0 else S
            k_start = max(0, i - w2) if w2 > 0 else 0
            k_end = i + 1 if w2 > 0 else S

            q_i = q[:, i:i+1, :, :]  # [B, 1, H, D]
            k1_w = k1[:, j_start:j_end, :, :]
            k2_w = k2[:, k_start:k_end, :, :]
            v1_w = v1[:, j_start:j_end, :, :]
            v2_w = v2[:, k_start:k_end, :, :]
            dO_i = grad_output[:, i:i+1, :, :]

            # Recompute logits e attn per questa posizione
            logits = torch.einsum("bthd,bshd,brhd->bh tsr", q_i, k1_w, k2_w) * sm_scale
            logits = logits.squeeze(2)  # [B, H, w1, w2]
            shape = logits.shape
            logits_flat = logits.reshape(B, H, -1)
            attn = F.softmax(logits_flat, dim=-1).reshape(shape)

            # Squeeza le dimensioni singole: [B, 1, H, D] → [B, H, D]
            dO_i_sq = dO_i.squeeze(1)
            q_i_sq = q_i.squeeze(1)  # [B, H, D]

            # dv1[b, j, h, d] = Σ_k attn[b, h, j, k] · dO_i[b, h, d] · v2_w[b, k, h, d]
            dv1[:, j_start:j_end, :, :] += torch.einsum("bhjk,bhd,bkhd->bjhd", attn, dO_i_sq, v2_w)

            # dv2[b, k, h, d] = Σ_j attn[b, h, j, k] · dO_i[b, h, d] · v1_w[b, j, h, d]
            dv2[:, k_start:k_end, :, :] += torch.einsum("bhjk,bhd,bjhd->bkhd", attn, dO_i_sq, v1_w)

            # dp[b, h, j, k] = dO_i[b, h, d] · v1_w[b, j, h, d] · v2_w[b, k, h, d]
            dp = torch.einsum("bhd,bjhd,bkhd->bhjk", dO_i_sq, v1_w, v2_w)
            dp_flat = dp.reshape(B, H, -1)
            attn_flat = attn.reshape(B, H, -1)
            ds_flat = attn_flat * (dp_flat - (attn_flat * dp_flat).sum(dim=-1, keepdim=True))
            ds = ds_flat.reshape(*attn.shape)

            # dq[b, h, d] = Σ_j Σ_k ds[b, h, j, k] · k1_w[b, j, h, d] · k2_w[b, k, h, d]
            dq[:, i, :, :] += torch.einsum("bhjk,bjhd,bkhd->bhd", ds, k1_w, k2_w) * sm_scale

            # dk1[b, j, h, d] = Σ_k ds[b, h, j, k] · q_i[b, h, d] · k2_w[b, k, h, d]
            dk1[:, j_start:j_end, :, :] += torch.einsum("bhjk,bhd,bkhd->bjhd", ds, q_i_sq, k2_w) * sm_scale

            # dk2[b, k, h, d] = Σ_j ds[b, h, j, k] · q_i[b, h, d] · k1_w[b, j, h, d]
            dk2[:, k_start:k_end, :, :] += torch.einsum("bhjk,bhd,bjhd->bkhd", ds, q_i_sq, k1_w) * sm_scale

        return dq, dk1, dk2, dv1, dv2, None, None


# ======================= Forward passes. =======================
def torch_fwd_ref(
    q,
    k1,
    k2,
    v1,
    v2,
    w1=-1,
    w2=-1,
    k2_bias=None,
    v2_bias=None,
    variant=None,
    use_fp32=True,
    sm_scale=None,
    disable_kv_bias=False,
):
    """Reference impl for simplicial attention.
    w1, w2: Int. Positive values for block-local windowed attention.
    variant: [None, strassen, rank1]. Algorithm to compute logits.

    Given [batch, sequence, local_heads, embedding] inputs,
    forms a attention probability cube [batch, sequence, sequence, sequence]
    where each logits[b, n, i, j, k] = sum_h(q[b, i, n, h] * k1[b, j, n, h] * k2[b, k, n, h])
    for i - w1 < j <= i and i - w2 < k <= i

    Then for each i, gets softmax across j, k
    And uses probabilities to lookup against values:
    output[b, i, n, h] = sum_jk(prob[b, n, i, j, k] * v1[b, j, n, h] * v2[b, k, n, h])
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    b, s, k, d = q.shape
    if use_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        q = q.to(torch.float32)
        k1 = k1.to(torch.float32)
        k2 = k2.to(torch.float32)
        v1 = v1.to(torch.float32)
        v2 = v2.to(torch.float32)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    if not k2_bias:
        k2_bias = 1.0 / d
    if not v2_bias:
        v2_bias = 1.0

    if not disable_kv_bias:
        k2 = k2 + k2_bias
        v2 = v2 + v2_bias

    if variant == "strassen":
        qk1 = torch.einsum("btkh,bskh->bkts", q, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q, k2)
        k1k2 = torch.einsum(
            "bskh,brkh->bksr", k1, k2
        )  # note this is a matmul, not kronecker like in normal two_simplicial.

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3) + k1k2.unsqueeze(2)
    if variant == "rank1":
        qk1 = torch.einsum("btkh,bskh->bkts", q, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q, k2)

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3)
    else:  # Default triple product.
        logits = torch.einsum("btkh,bskh,brkh->bktsr", q, k1, k2)

    # If block local.
    if w1 > 0 and w2 > 0:
        # Create [q, kv1, kv2] shaped mask.
        q_idx = torch.arange(s)[:, None, None].to(
            device=torch.accelerator.current_accelerator()
        )
        kv1_idx = torch.arange(s)[None, :, None].to(
            device=torch.accelerator.current_accelerator()
        )
        kv2_idx = torch.arange(s)[None, None, :].to(
            device=torch.accelerator.current_accelerator()
        )

        kv1_mask = ((q_idx - w1) < kv1_idx) & (kv1_idx <= q_idx)
        kv2_mask = ((q_idx - w2) < kv2_idx) & (kv2_idx <= q_idx)
        local_mask = kv1_mask & kv2_mask
        logits = torch.where(local_mask[None, None, :], logits, -float("inf"))

    # Per q token softmax. [b, k, t, t, t] -> [b, k, t, t*t]
    logits = logits * sm_scale
    shape = logits.shape
    logits_reshaped = logits.view(*shape[:-2], -1)
    attn_prob = F.softmax(logits_reshaped, dim=-1).type_as(q)
    attn_prob = attn_prob.view(*shape)
    output = torch.einsum("bktsr,bskh,brkh->btkh", attn_prob, v1, v2)

    return output


# ======================= Backward passes. =======================
def torch_bwd_ref(
    q, k1, k2, v1, v2, w1, w2, d_output, variant=None, use_norm=True, use_fp32=False
):
    """Two Simplicial attn bwd pass via autograd."""
    _, seq_len, _, _ = q.shape
    torch.backends.cuda.matmul.allow_tf32 = True
    if use_fp32:
        dtype = torch.float64
        torch.backends.cuda.matmul.allow_tf32 = False
        q = q.clone().to(dtype)
        k1 = k1.clone().to(dtype)
        k2 = k2.clone().to(dtype)
        v1 = v1.clone().to(dtype)
        v2 = v2.clone().to(dtype)
        d_output = d_output.clone().to(dtype)
    q.requires_grad_()
    k1.requires_grad_()
    k2.requires_grad_()
    v1.requires_grad_()
    v2.requires_grad_()

    if use_norm:
        k2_norm = k2 + 1.0 / k2.shape[-1]
        v2_norm = v2 + 1.0
        q_norm = q / math.sqrt(q.shape[-1])
    else:
        k2_norm = k2
        v2_norm = v2
        q_norm = q

    if variant == "strassen":
        qk1 = torch.einsum("btkh,bskh->bkts", q_norm, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q_norm, k2_norm)
        k1k2 = torch.einsum(
            "bskh,brkh->bksr", k1, k2_norm
        )  # note this is a matmul, not kronecker like in normal two_simplicial.

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3) + k1k2.unsqueeze(2)
    if variant == "rank1":
        qk1 = torch.einsum("btkh,bskh->bkts", q_norm, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q_norm, k2_norm)

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3)
    else:
        logits = torch.einsum("btkh,bskh,brkh->bktsr", q_norm, k1, k2_norm)

    if w1 > 0 and w2 > 0:
        q_idx = torch.arange(seq_len)[:, None, None].to(
            device=torch.accelerator.current_accelerator()
        )
        kv1_idx = torch.arange(seq_len)[None, :, None].to(
            device=torch.accelerator.current_accelerator()
        )
        kv2_idx = torch.arange(seq_len)[None, None, :].to(
            device=torch.accelerator.current_accelerator()
        )

        kv1_mask = ((q_idx - w1) < kv1_idx) & (kv1_idx <= q_idx)
        kv2_mask = ((q_idx - w2) < kv2_idx) & (kv2_idx <= q_idx)

        local_mask = kv1_mask & kv2_mask
        logits = torch.where(local_mask[None, None, :], logits, -float("inf"))

    # Per q token softmax. [b, k, t, t, t] -> [b, k, t, t*t]
    shape = logits.shape
    logits_reshaped = logits.view(*shape[:-2], -1)
    attn_prob = F.softmax(logits_reshaped, dim=-1).type_as(q)
    attn_prob = attn_prob.view(*shape)

    # Perform the einsum operation
    output = torch.einsum("bktsr,bskh,brkh->btkh", attn_prob, v1, v2_norm)

    output.backward(d_output)
    dv1 = v1.grad.clone()
    dv2 = v2.grad.clone()
    dk1 = k1.grad.clone()
    dk2 = k2.grad.clone()
    dq = q.grad.clone()

    # dp_expected = attn_prob.grad.clone()
    return dq, dk1, dk2, dv1, dv2, output.to(torch.float32)


def torch_simplicial_bwd(
    q,
    k1,
    k2,
    v1,
    v2,
    output,
    d_output,
    variant=None,
    use_norm=True,
    compute_dtype=torch.bfloat16,
):
    """Manual two_simplicial backward pass."""
    torch.backends.cuda.matmul.allow_tf32 = True
    q = q.to(dtype=compute_dtype).detach()
    k1 = k1.to(dtype=compute_dtype).detach()
    k2 = k2.to(dtype=compute_dtype).detach()
    v1 = v1.to(dtype=compute_dtype).detach()
    v2 = v2.to(dtype=compute_dtype).detach()
    d_output = d_output.to(dtype=compute_dtype).detach()

    bs, seq_len, num_heads, head_dim = q.shape
    if use_norm:
        softmax_scale = head_dim**-0.5
        k2 += 1 / head_dim
        v2 += 1

    else:
        softmax_scale = 1.0
    if variant == "strassen":
        qk1 = torch.einsum("btkh,bskh->bkts", q * softmax_scale, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q * softmax_scale, k2)
        k1k2 = torch.einsum(
            "bskh,brkh->bksr", k1, k2
        )  # note this is a matmul, not kronecker like in normal two_simplicial.

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3) + k1k2.unsqueeze(2)
    elif variant == "rank1":
        qk1 = torch.einsum("btkh,bskh->bkts", q * softmax_scale, k1)
        qk2 = torch.einsum("btkh,brkh->bktr", q * softmax_scale, k2)

        logits = qk1.unsqueeze(4) + qk2.unsqueeze(3)
    else:
        logits = torch.einsum("btkh,bskh,brkh->bktsr", q * softmax_scale, k1, k2)

    # Per q token softmax. [b, k, t, t, t] -> [b, k, t, t*t]
    shape = logits.shape
    logits_reshaped = logits.view(*shape[:-2], -1)
    attn_prob = F.softmax(logits_reshaped, dim=-1).type_as(q)
    attn_prob = attn_prob.view(*shape)

    dv1 = torch.einsum("bktsr,btkh,brkh->bskh", attn_prob, d_output, v2)
    dv2 = torch.einsum("bktsr,btkh,bskh->brkh", attn_prob, d_output, v1)

    dp = torch.einsum("btkh,bskh,brkh->bktsr", d_output, v1, v2)
    shape = dp.shape
    dp = dp.reshape(*shape[:-2], -1)

    prob = attn_prob.view(*shape[:-2], -1)
    ds = (
        (torch.diag_embed(prob) - prob.unsqueeze(-1) * prob.unsqueeze(-2))
        @ dp.unsqueeze(-1)
    ).squeeze(-1)
    ds = ds.reshape(shape)  # [b, k, t, s, r]

    if variant == "strassen":
        k1_plus_k2 = k1[:, :, None, :, :] + k2[:, None, :, :, :]
        q_plus_k2 = q[:, :, None, :, :] * softmax_scale + k2[:, None, :, :, :]
        q_plus_k1 = q[:, :, None, :, :] * softmax_scale + k1[:, None, :, :, :]
        dq = torch.einsum("bktsr,bsrkh->btkh", ds, k1_plus_k2) * softmax_scale
        dk1 = torch.einsum("bktsr,btrkh->bskh", ds, q_plus_k2)
        dk2 = torch.einsum("bktsr,btskh->brkh", ds, q_plus_k1)
    elif variant == "rank1":
        k1_plus_k2 = k1[:, :, None, :, :] + k2[:, None, :, :, :]
        dq = torch.einsum("bktsr,bsrkh->btkh", ds, k1_plus_k2) * softmax_scale
        dk1 = torch.einsum("bktsr,btkh->bskh", ds, q * softmax_scale)
        dk2 = torch.einsum("bktsr,btkh->brkh", ds, q * softmax_scale)
    else:
        dq = torch.einsum("bktsr,bskh,brkh->btkh", ds, k1, k2) * softmax_scale
        dk1 = torch.einsum("bktsr,btkh,brkh->bskh", ds, q, k2) * softmax_scale
        dk2 = torch.einsum("bktsr,btkh,bskh->brkh", ds, q, k1) * softmax_scale

    dq = dq.detach()
    dk1 = dk1.detach()
    dk2 = dk2.detach()
    dv1 = dv1.detach()
    dv2 = dv2.detach()

    return dq, dk1, dk2, dv1, dv2