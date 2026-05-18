# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _run_indexed_accum(
    *,
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    blocked: bool,
    candidate_offset: int,
    block_c: int,
    block_heads: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
        accumulate_indexed_sparse_mla_attention_chunk,
    )

    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM",
        "1" if blocked else "0",
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C", str(block_c))
    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS",
        str(block_heads),
    )
    max_score = torch.full(
        (q.shape[0], q.shape[1]),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    denom = torch.zeros(
        (q.shape[0], q.shape[1]),
        device=q.device,
        dtype=torch.float32,
    )
    acc = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )
    accumulate_indexed_sparse_mla_attention_chunk(
        q,
        kv_flat,
        indices,
        lens,
        scale,
        max_score,
        denom,
        acc,
        candidate_offset=candidate_offset,
    )
    torch.cuda.synchronize()
    return max_score, denom, acc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "num_heads", "head_dim", "num_candidates", "block_c", "block_heads"),
    [
        (torch.float16, 10, 512, 20, 16, 8),
        (torch.float16, 16, 128, 48, 32, 4),
        (torch.bfloat16, 10, 512, 20, 16, 8),
    ],
)
def test_indexed_sparse_mla_blocked_accum_matches_legacy(
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    num_candidates: int,
    block_c: int,
    block_heads: int,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 4
    candidate_offset = 16
    kv_tokens = max(64, num_candidates + 16)
    scale = 0.17

    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_flat = torch.randn(kv_tokens, head_dim, device=device, dtype=dtype)
    indices = torch.full(
        (num_tokens, num_candidates),
        -1,
        device=device,
        dtype=torch.int32,
    )
    indices[0, :4] = torch.tensor([0, 3, 7, 11], device=device)
    indices[1, : min(18, num_candidates)] = torch.arange(
        2,
        2 + min(18, num_candidates),
        device=device,
    )
    indices[2, :6] = torch.tensor([1, 4, -1, 6, 8, -1], device=device)
    # Token 3 exercises an all-invalid block via -1 sentinels.
    lens = torch.tensor(
        [
            candidate_offset,
            candidate_offset + min(18, num_candidates),
            candidate_offset + 5,
            candidate_offset + num_candidates,
        ],
        device=device,
        dtype=torch.int32,
    )

    def run(blocked: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _run_indexed_accum(
            q=q,
            kv_flat=kv_flat,
            indices=indices,
            lens=lens,
            scale=scale,
            blocked=blocked,
            candidate_offset=candidate_offset,
            block_c=block_c,
            block_heads=block_heads,
            monkeypatch=monkeypatch,
        )

    legacy_max, legacy_denom, legacy_acc = run(False)
    blocked_max, blocked_denom, blocked_acc = run(True)

    assert torch.isfinite(blocked_denom).all()
    assert torch.isfinite(blocked_acc).all()
    assert torch.all(blocked_denom[0] == 0)
    assert torch.all(blocked_denom[3] == 0)
    torch.testing.assert_close(blocked_max, legacy_max, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(blocked_denom, legacy_denom, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(blocked_acc, legacy_acc, rtol=6e-3, atol=6e-3)

    legacy_out = legacy_acc / legacy_denom[..., None].clamp_min(1.0e-20)
    blocked_out = blocked_acc / blocked_denom[..., None].clamp_min(1.0e-20)
    torch.testing.assert_close(blocked_out, legacy_out, rtol=6e-3, atol=6e-3)
