# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_integration as b12x_dsv4


def _enable_b12x_path(monkeypatch: pytest.MonkeyPatch, path_name: str) -> None:
    monkeypatch.setattr(b12x_dsv4.envs, "VLLM_USE_B12X_DEEPSEEK_V4", True)
    monkeypatch.setattr(
        b12x_dsv4.envs,
        f"VLLM_USE_B12X_DEEPSEEK_V4_{path_name}",
        True,
    )
    monkeypatch.setattr(b12x_dsv4, "_is_sm12x_cuda", lambda: True)
    monkeypatch.setattr(b12x_dsv4, "_b12x_available", lambda _path_name: True)
    monkeypatch.setattr(
        b12x_dsv4,
        "logger",
        SimpleNamespace(
            info_once=lambda *args, **kwargs: None,
            warning_once=lambda *args, **kwargs: None,
        ),
    )


def _fake_b12x_integration(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_b12x = ModuleType("b12x")
    fake_integration = ModuleType("b12x.integration")
    fake_b12x.integration = fake_integration
    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    return fake_integration


def _padded_compressed_page_cache(
    *,
    pages: int,
    page_size: int,
) -> torch.Tensor:
    page_nbytes = b12x_dsv4._compressed_mla_page_nbytes(page_size)
    storage = torch.empty((pages, page_nbytes), dtype=torch.uint8)
    return torch.as_strided(
        storage,
        size=(pages, page_size, b12x_dsv4._COMPRESSED_MLA_TOKEN_BYTES),
        stride=(page_nbytes, b12x_dsv4._COMPRESSED_MLA_TOKEN_BYTES, 1),
    )


def test_b12x_deepseek_v4_paths_remain_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b12x_dsv4.envs, "VLLM_USE_B12X_DEEPSEEK_V4", True)
    monkeypatch.setattr(b12x_dsv4.envs, "VLLM_USE_B12X_DEEPSEEK_V4_MHC", None)
    monkeypatch.setattr(b12x_dsv4.envs, "VLLM_USE_B12X_DEEPSEEK_V4_INDEXER", None)
    monkeypatch.setattr(
        b12x_dsv4.envs,
        "VLLM_USE_B12X_DEEPSEEK_V4_COMPRESSED_MLA",
        None,
    )

    assert b12x_dsv4._enabled("MHC")
    assert not b12x_dsv4._enabled("INDEXER")
    assert not b12x_dsv4._enabled("COMPRESSED_MLA")


def test_paged_mqa_indexer_flattens_deepseek_decode_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_b12x_path(monkeypatch, "INDEXER")
    b12x_integration = _fake_b12x_integration(monkeypatch)

    rows = 4
    heads = 3
    topk = 2
    q_fp8 = torch.empty((2, 2, heads, 128), dtype=torch.float8_e4m3fn)
    weights = torch.ones((rows, heads), dtype=torch.float32)
    index_k_cache = torch.empty(
        (
            5,
            b12x_dsv4._INDEX_PAGE_SIZE,
            b12x_dsv4._INDEX_TOKEN_BYTES,
        ),
        dtype=torch.uint8,
    )
    seq_lens = torch.tensor([[64, 128], [192, 256]], dtype=torch.int32)
    block_table = torch.tensor(
        [[4, 5, 6, 7], [10, 11, 12, 13]],
        dtype=torch.int32,
    )
    topk_indices = torch.full((rows, topk), -1, dtype=torch.int32)

    captured: dict[str, object] = {}

    def fake_workspace(**kwargs):
        captured["workspace_kwargs"] = kwargs
        return SimpleNamespace(name="workspace")

    def fake_prepare(**kwargs):
        captured["metadata_kwargs"] = kwargs
        return SimpleNamespace(
            real_page_table=kwargs["real_page_table"],
            cache_seqlens_int32=kwargs["cache_seqlens_int32"],
            expected_num_q_heads=kwargs["expected_num_q_heads"],
        )

    def fake_topk(**kwargs):
        captured["topk_kwargs"] = kwargs
        out = kwargs["out_indices"]
        out.copy_(torch.tensor([[3, 2], [7, 6], [11, 10], [15, 14]], dtype=torch.int32))
        return out

    monkeypatch.setattr(b12x_dsv4, "_get_paged_indexer_workspace", fake_workspace)
    monkeypatch.setattr(
        b12x_integration,
        "prepare_paged_mqa_indexer_metadata",
        fake_prepare,
        raising=False,
    )
    monkeypatch.setattr(
        b12x_integration,
        "paged_mqa_index_decode_supertile_topk_fp8",
        fake_topk,
        raising=False,
    )

    assert b12x_dsv4.b12x_paged_mqa_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        topk_indices=topk_indices,
        topk=topk,
    )

    metadata_kwargs = captured["metadata_kwargs"]
    torch.testing.assert_close(
        metadata_kwargs["real_page_table"],
        block_table.repeat_interleave(2, dim=0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        metadata_kwargs["cache_seqlens_int32"],
        seq_lens.reshape(-1),
        rtol=0,
        atol=0,
    )
    assert metadata_kwargs["expected_num_q_heads"] == heads
    assert metadata_kwargs["build_schedule"] is False
    assert metadata_kwargs["validate_raw_lengths"] is False

    workspace_kwargs = captured["workspace_kwargs"]
    assert workspace_kwargs["rows"] == rows
    assert workspace_kwargs["heads"] == heads
    assert workspace_kwargs["topk"] == topk
    assert workspace_kwargs["page_width"] == block_table.shape[1]

    topk_kwargs = captured["topk_kwargs"]
    assert topk_kwargs["q_fp8"].shape == (rows, heads, 128)
    assert topk_kwargs["index_k_cache"].shape == (
        5,
        b12x_dsv4._INDEX_PAGE_SIZE * b12x_dsv4._INDEX_TOKEN_BYTES,
    )
    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[3, 2], [7, 6], [11, 10], [15, 14]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_compressed_mla_decode_forwards_padded_page_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_b12x_path(monkeypatch, "COMPRESSED_MLA")
    b12x_integration = _fake_b12x_integration(monkeypatch)

    rows = 2
    heads = 3
    swa_page_size = 128
    indexed_page_size = 2
    q = torch.randn((rows, 1, heads, 512), dtype=torch.bfloat16)
    swa_cache = _padded_compressed_page_cache(pages=2, page_size=swa_page_size)
    indexed_cache = torch.empty(
        (
            3,
            b12x_dsv4._compressed_mla_page_nbytes(indexed_page_size),
        ),
        dtype=torch.uint8,
    )
    swa_indices = torch.tensor([[[0, 1, 2]], [[3, 4, 5]]], dtype=torch.int32)
    indexed_indices = torch.tensor([[[10, 11]], [[12, 13]]], dtype=torch.int32)
    swa_lens = torch.tensor([3, 2], dtype=torch.int32)
    indexed_lens = torch.tensor([2, 1], dtype=torch.int32)
    attn_sink = torch.arange(heads + 1, dtype=torch.float32)
    output = torch.full((rows, heads + 1, 512), 9, dtype=torch.bfloat16)
    expected = torch.randn((rows, heads, 512), dtype=torch.bfloat16)

    captured: dict[str, object] = {}

    def fake_workspace(**kwargs):
        captured["workspace_kwargs"] = kwargs
        return SimpleNamespace(name="workspace")

    def fake_decode_forward(**kwargs):
        captured["decode_kwargs"] = kwargs
        return expected

    monkeypatch.setattr(b12x_dsv4, "_get_compressed_mla_workspace", fake_workspace)
    monkeypatch.setattr(
        b12x_integration,
        "compressed_mla_decode_forward",
        fake_decode_forward,
        raising=False,
    )

    assert b12x_dsv4.b12x_compressed_mla_decode(
        q=q,
        compressed_k_cache=indexed_cache,
        swa_k_cache=swa_cache,
        topk_indices=indexed_indices,
        topk_lens=indexed_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        compressed_page_size=indexed_page_size,
        swa_page_size=swa_page_size,
        sm_scale=0.125,
        attn_sink=attn_sink,
        num_heads=heads,
        output=output,
    )

    workspace_kwargs = captured["workspace_kwargs"]
    assert workspace_kwargs["rows"] == rows
    assert workspace_kwargs["heads"] == heads
    assert workspace_kwargs["topk"] == swa_indices.shape[-1] + indexed_indices.shape[-1]

    decode_kwargs = captured["decode_kwargs"]
    assert decode_kwargs["q_all"].shape == (rows, heads, 512)
    assert decode_kwargs["swa_k_cache"].shape == (
        2,
        b12x_dsv4._compressed_mla_page_nbytes(swa_page_size),
    )
    assert decode_kwargs["indexed_k_cache"].shape == indexed_cache.shape
    torch.testing.assert_close(decode_kwargs["swa_indices"], swa_indices[:, 0, :])
    torch.testing.assert_close(
        decode_kwargs["indexed_indices"],
        indexed_indices[:, 0, :],
    )
    torch.testing.assert_close(decode_kwargs["attn_sink"], attn_sink[:heads])

    torch.testing.assert_close(output[:, :heads, :], expected)
    torch.testing.assert_close(output[:, heads:, :], torch.zeros_like(output[:, heads:, :]))
