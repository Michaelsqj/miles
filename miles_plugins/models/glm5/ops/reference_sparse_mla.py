"""Correctness fallback for GLM-5 sparse MLA backward."""

import torch


def reference_sparse_mla_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute sparse-attention gradients with ordinary PyTorch operations.

    This is intentionally a slow fallback. Callers should use it only after the
    fused backward has returned a non-finite value.
    """
    seq_len, num_heads, qk_dim = q.shape
    seq_len_kv, kv_groups, kv_dim = kv.shape
    value_dim = out.shape[-1]
    if qk_dim != kv_dim:
        raise ValueError(f"q/kv dimensions differ: {qk_dim} != {kv_dim}")
    if num_heads % kv_groups:
        raise ValueError(f"{num_heads} heads are not divisible by {kv_groups} KV groups")
    if indices.shape[:2] != (seq_len, kv_groups):
        raise ValueError(f"unexpected indices shape {tuple(indices.shape)}")

    heads_per_group = num_heads // kv_groups
    dq = torch.zeros_like(q, dtype=torch.float32)
    dkv = torch.zeros_like(kv, dtype=torch.float32)

    for group in range(kv_groups):
        head_start = group * heads_per_group
        head_end = head_start + heads_per_group
        kv_group = kv[:, group].float()

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            selected_indices = indices[start:end, group]
            valid = selected_indices >= 0
            safe_indices = selected_indices.clamp(min=0, max=seq_len_kv - 1)
            selected_kv = kv_group.index_select(0, safe_indices.reshape(-1)).reshape(
                end - start, selected_indices.shape[-1], kv_dim
            )

            q_chunk = q[start:end, head_start:head_end].float()
            grad_out_chunk = grad_out[start:end, head_start:head_end].float()
            logits = torch.einsum("chd,ckd->chk", q_chunk, selected_kv) * sm_scale
            logits = logits.masked_fill(~valid.unsqueeze(1), torch.finfo(logits.dtype).min)
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities * valid.unsqueeze(1)
            probability_sum = probabilities.sum(dim=-1, keepdim=True)
            probabilities = torch.where(
                probability_sum > 0,
                probabilities / probability_sum.clamp_min(torch.finfo(probabilities.dtype).tiny),
                torch.zeros_like(probabilities),
            )

            delta = (
                out[start:end, head_start:head_end].float() * grad_out_chunk
            ).sum(dim=-1, keepdim=True)
            grad_probability = torch.einsum(
                "chd,ckd->chk", grad_out_chunk, selected_kv[..., :value_dim]
            )
            grad_score = probabilities * (grad_probability - delta) * sm_scale
            dq[start:end, head_start:head_end] = torch.einsum(
                "chk,ckd->chd", grad_score, selected_kv
            )

            grad_selected_kv = torch.einsum("chk,chd->ckd", grad_score, q_chunk)
            grad_selected_kv[..., :value_dim] += torch.einsum(
                "chk,chd->ckd", probabilities, grad_out_chunk
            )
            grad_selected_kv *= valid.unsqueeze(-1)
            dkv[:, group].index_add_(
                0,
                safe_indices.reshape(-1),
                grad_selected_kv.reshape(-1, kv_dim),
            )

    return dq.to(q.dtype), dkv.to(kv.dtype)
