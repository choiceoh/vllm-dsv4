# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _reference_indexed_accum(
    *,
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    candidate_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    q_f32 = q.float()
    kv_f32 = kv_flat.float()
    for token_idx in range(q.shape[0]):
        local_eff = min(
            indices.shape[1],
            max(int(lens[token_idx].item()) - candidate_offset, 0),
        )
        if local_eff <= 0:
            continue
        kv_indices = indices[token_idx, :local_eff]
        kv_indices = kv_indices[kv_indices >= 0]
        if kv_indices.numel() == 0:
            continue
        kv = kv_f32[kv_indices.long()]
        scores = torch.matmul(q_f32[token_idx], kv.T) * scale
        token_max = scores.max(dim=1).values
        weights = torch.exp(scores - token_max[:, None])
        max_score[token_idx] = token_max
        denom[token_idx] = weights.sum(dim=1)
        acc[token_idx] = torch.matmul(weights, kv)
    return max_score, denom, acc


def _run_indexed_accum(
    *,
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    blocked: bool,
    fp32_value: bool,
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
    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE",
        "1" if fp32_value else "0",
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C", str(block_c))
    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS",
        str(block_heads),
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS", "4")
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES", "2")
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
        (torch.float16, 10, 512, 48, 32, 8),
        (torch.float16, 10, 512, 48, 32, 4),
        (torch.float16, 16, 128, 48, 32, 4),
        (torch.bfloat16, 10, 512, 20, 16, 8),
        (torch.bfloat16, 10, 512, 48, 32, 8),
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
    row1_valid = min(48, num_candidates)
    indices[1, :row1_valid] = torch.arange(
        2,
        2 + row1_valid,
        device=device,
    )
    indices[2, :6] = torch.tensor([1, 4, -1, 6, 8, -1], device=device)
    # Token 3 exercises an all-invalid block via -1 sentinels.
    lens = torch.tensor(
        [
            candidate_offset,
            candidate_offset + row1_valid,
            candidate_offset + 5,
            candidate_offset + num_candidates,
        ],
        device=device,
        dtype=torch.int32,
    )

    def run(
        blocked: bool,
        fp32_value: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _run_indexed_accum(
            q=q,
            kv_flat=kv_flat,
            indices=indices,
            lens=lens,
            scale=scale,
            blocked=blocked,
            fp32_value=fp32_value,
            candidate_offset=candidate_offset,
            block_c=block_c,
            block_heads=block_heads,
            monkeypatch=monkeypatch,
        )

    legacy_max, legacy_denom, legacy_acc = run(False)
    blocked_max, blocked_denom, blocked_acc = run(True)
    fp32_max, fp32_denom, fp32_acc = run(True, fp32_value=True)

    for test_max, test_denom, test_acc in (
        (blocked_max, blocked_denom, blocked_acc),
        (fp32_max, fp32_denom, fp32_acc),
    ):
        assert torch.isfinite(test_denom).all()
        assert torch.isfinite(test_acc).all()
        assert torch.all(test_denom[0] == 0)
        assert torch.all(test_denom[3] == 0)
        torch.testing.assert_close(test_max, legacy_max, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(test_denom, legacy_denom, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(test_acc, legacy_acc, rtol=6e-3, atol=6e-3)

    legacy_out = legacy_acc / legacy_denom[..., None].clamp_min(1.0e-20)
    for test_acc, test_denom in (
        (blocked_acc, blocked_denom),
        (fp32_acc, fp32_denom),
    ):
        test_out = test_acc / test_denom[..., None].clamp_min(1.0e-20)
        torch.testing.assert_close(test_out, legacy_out, rtol=6e-3, atol=6e-3)


@pytest.mark.parametrize(
    (
        "requested_head_block",
        "candidate_block",
        "block_d",
        "num_heads",
        "expected",
    ),
    [
        (8, 32, 512, 10, 4),
        (8, 32, 512, 4, 4),
        (8, 32, 256, 10, 8),
        (8, 16, 512, 10, 8),
        (0, 32, 512, 10, 0),
        (8, 64, 512, 10, 0),
    ],
)
def test_prefill_blocked_indexed_accum_head_block(
    requested_head_block: int,
    candidate_block: int,
    block_d: int,
    num_heads: int,
    expected: int,
):
    from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
        _prefill_blocked_indexed_accum_head_block,
    )

    assert (
        _prefill_blocked_indexed_accum_head_block(
            requested_head_block=requested_head_block,
            candidate_block=candidate_block,
            block_d=block_d,
            num_heads=num_heads,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("env_name", "value", "function_name", "expected"),
    [
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C", "16", "block_c", 16),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C", "32", "block_c", 32),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C", "8", "block_c", 0),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS", "1", "block_heads", 1),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS", "2", "block_heads", 2),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS", "4", "block_heads", 4),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS", "8", "block_heads", 8),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS", "16", "block_heads", 0),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS", "4", "block_warps", 4),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS", "8", "block_warps", 8),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS", "2", "block_warps", 4),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES", "2", "block_stages", 2),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES", "3", "block_stages", 3),
        ("VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES", "4", "block_stages", 2),
    ],
)
def test_prefill_blocked_accum_env_validation(
    env_name: str,
    value: str,
    function_name: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.backends.mla import sparse_mla_env

    functions = {
        "block_c": sparse_mla_env.triton_sparse_mla_prefill_block_c,
        "block_heads": sparse_mla_env.triton_sparse_mla_prefill_block_heads,
        "block_warps": sparse_mla_env.triton_sparse_mla_prefill_block_warps,
        "block_stages": sparse_mla_env.triton_sparse_mla_prefill_block_stages,
    }

    monkeypatch.setenv(env_name, value)
    assert functions[function_name]() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_indexed_sparse_mla_blocked_accum_c32_second_tile_matches_reference(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    num_tokens = 2
    num_heads = 10
    head_dim = 512
    num_candidates = 64
    candidate_offset = 7
    scale = 1.0

    q = torch.zeros(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    kv_flat = torch.zeros(96, head_dim, device=device, dtype=dtype)
    q[:, :, 0] = 1
    kv_flat[:num_candidates, 0] = torch.linspace(
        -3.0,
        3.0,
        num_candidates,
        device=device,
        dtype=dtype,
    )
    indices = torch.arange(num_candidates, device=device, dtype=torch.int32).expand(
        num_tokens,
        num_candidates,
    ).contiguous()
    lens = torch.tensor(
        [
            candidate_offset + num_candidates,
            candidate_offset + 48,
        ],
        device=device,
        dtype=torch.int32,
    )

    ref_max, ref_denom, ref_acc = _reference_indexed_accum(
        q=q,
        kv_flat=kv_flat,
        indices=indices,
        lens=lens,
        scale=scale,
        candidate_offset=candidate_offset,
    )
    blocked_max, blocked_denom, blocked_acc = _run_indexed_accum(
        q=q,
        kv_flat=kv_flat,
        indices=indices,
        lens=lens,
        scale=scale,
        blocked=True,
        fp32_value=True,
        candidate_offset=candidate_offset,
        block_c=32,
        block_heads=8,
        monkeypatch=monkeypatch,
    )

    assert torch.all(ref_max == kv_flat[lens - candidate_offset - 1, 0][:, None])
    torch.testing.assert_close(blocked_max, ref_max, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(blocked_denom, ref_denom, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(blocked_acc, ref_acc, rtol=6e-3, atol=6e-3)
