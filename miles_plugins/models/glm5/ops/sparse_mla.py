import logging

import torch

from .reference_sparse_mla import reference_sparse_mla_backward
from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface


logger = logging.getLogger(__name__)


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        grad_output = grad_output.contiguous()
        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, grad_output, indices, tl_lse, sm_scale=scaling)

        dq_finite = torch.isfinite(tl_dq)
        dkv_finite = torch.isfinite(tl_dkv)
        if not bool(dq_finite.all() and dkv_finite.all()):
            logger.error(
                "GLM-5 SparseMLA fused backward returned non-finite gradients; "
                "recomputing the call with the PyTorch correctness fallback "
                "(dq=%d, dkv=%d)",
                int((~dq_finite).sum()),
                int((~dkv_finite).sum()),
            )
            ref_dq, ref_dkv = reference_sparse_mla_backward(
                q,
                kv,
                tl_out,
                grad_output,
                indices,
                scaling,
            )
            tl_dq = torch.where(dq_finite, tl_dq, ref_dq)
            tl_dkv = torch.where(dkv_finite, tl_dkv, ref_dkv)

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
