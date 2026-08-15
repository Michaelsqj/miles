import torch

from miles_plugins.models.glm5.ops.reference_sparse_mla import reference_sparse_mla_backward


def test_reference_sparse_mla_backward_matches_autograd() -> None:
    torch.manual_seed(7)
    seq_len = 5
    heads = 2
    value_dim = 4
    qk_dim = 6
    scale = qk_dim**-0.5

    q = torch.randn(seq_len, heads, qk_dim, requires_grad=True)
    kv = torch.randn(seq_len, 1, qk_dim, requires_grad=True)
    indices = torch.tensor(
        [
            [[0, -1, -1, -1]],
            [[0, 1, -1, -1]],
            [[0, 1, 2, -1]],
            [[0, 1, 2, 3]],
            [[1, 2, 3, 4]],
        ],
        dtype=torch.int32,
    )
    outputs = []
    for token in range(seq_len):
        token_indices = indices[token, 0]
        token_indices = token_indices[token_indices >= 0].long()
        selected = kv[token_indices, 0]
        probabilities = torch.softmax(q[token] @ selected.T * scale, dim=-1)
        outputs.append(probabilities @ selected[:, :value_dim])
    out = torch.stack(outputs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    ref_dq, ref_dkv = reference_sparse_mla_backward(
        q.detach(),
        kv.detach(),
        out.detach(),
        grad_out,
        indices,
        scale,
        chunk_size=2,
    )

    torch.testing.assert_close(ref_dq, q.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ref_dkv, kv.grad, atol=2e-5, rtol=2e-5)
