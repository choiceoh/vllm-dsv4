# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels used by the local DeepSeek V4 path."""

import os

import torch

from vllm.triton_utils import tl, triton

_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS = 8192


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_choice(name: str, default: int, choices: tuple[int, ...]) -> int:
    value = _env_int(name, default)
    if value in choices:
        return value
    return default


def _fp8_mqa_topk_stream_k_tiles_per_launch(seq_len_kv: int, topk: int) -> int:
    requested = _env_int("VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES", 1)
    if requested <= 1:
        return 1

    min_kv_tokens = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS",
        _SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS,
    )
    if seq_len_kv < max(topk * 16, min_kv_tokens * 2):
        return 1
    return min(requested, 2)


def _fp8_mqa_topk_stream_config(topk: int) -> tuple[int, int, int]:
    default_block_h = 4 if topk >= 2048 else 8
    default_block_d = 16 if topk >= 2048 else 32
    default_num_warps = 8
    topk_suffix = "TOPK2048" if topk >= 2048 else "TOPK512"

    block_h = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_BLOCK_H",
        default_block_h,
        (1, 2, 4, 8),
    )
    block_d = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_BLOCK_D",
        default_block_d,
        (16, 32, 64),
    )
    num_warps = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_NUM_WARPS",
        default_num_warps,
        (4, 8),
    )

    block_h = _env_int_choice(
        f"VLLM_SM12X_MQA_TOPK_TRITON_{topk_suffix}_BLOCK_H",
        block_h,
        (1, 2, 4, 8),
    )
    block_d = _env_int_choice(
        f"VLLM_SM12X_MQA_TOPK_TRITON_{topk_suffix}_BLOCK_D",
        block_d,
        (16, 32, 64),
    )
    num_warps = _env_int_choice(
        f"VLLM_SM12X_MQA_TOPK_TRITON_{topk_suffix}_NUM_WARPS",
        num_warps,
        (4, 8),
    )
    return block_h, block_d, num_warps


def _fp8_mqa_logits_config() -> tuple[int, int, int, int]:
    block_m = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_M",
        16,
        (8, 16, 32, 64),
    )
    block_n = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_N",
        128,
        (64, 128, 256),
    )
    block_d = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_D",
        64,
        (64, 128),
    )
    num_warps = _env_int_choice(
        "VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_NUM_WARPS",
        4,
        (4, 8),
    )
    return block_m, block_n, block_d, num_warps


def _fp8_mqa_logits_skip_invalid_n_tiles() -> bool:
    return bool(
        _env_int("VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_SKIP_INVALID_N_TILES", 0)
    )


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(
            f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions"
        )
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


@triton.jit(do_not_specialize=["num_q", "seq_len_kv"])
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q,
    seq_len_kv,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SKIP_INVALID_N_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (
        offs_n[None, :] < seq_end[:, None]
    )

    if SKIP_INVALID_N_TILES:
        tile_start_n = pid_n * BLOCK_N
        tile_end_n = tl.minimum(tile_start_n + BLOCK_N, seq_len_kv)
        row_has_tile = valid_m & (seq_start < tile_end_n) & (
            seq_end > tile_start_n
        )
        if tl.max(tl.where(row_has_tile, 1, 0), axis=0) == 0:
            store_mask = valid_m[:, None] & valid_n[None, :]
            tl.store(
                logits_ptr
                + offs_m[:, None] * stride_lm
                + offs_n[None, :] * stride_ln,
                tl.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=tl.float32),
                mask=store_mask,
            )
            return

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + d[None, :] * stride_kd,
                mask=valid_n[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, tl.trans(k), input_precision="tf32")
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(seq_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    logits = torch.empty(
        (num_q, seq_len_kv),
        device=q.device,
        dtype=torch.float32,
    )
    if num_q == 0 or seq_len_kv == 0:
        return logits

    block_m, block_n, block_d, num_warps = _fp8_mqa_logits_config()
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, block_n))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SKIP_INVALID_N_TILES=_fp8_mqa_logits_skip_invalid_n_tiles(),
        num_warps=num_warps,
    )
    return logits


@triton.jit
def _fp8_mqa_topk_stream_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    best_values_ptr,
    best_indices_ptr,
    tile_start,
    num_q,
    seq_len_kv,
    topk: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    K_TILES_PER_LAUNCH: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, topk)
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_q
    seq_start = tl.load(cu_seqlen_ks_ptr + row, mask=valid_row, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + row, mask=valid_row, other=0)

    min_i32: tl.constexpr = -2147483648
    invalid_key: tl.constexpr = 2147483647
    offs_merged = tl.arange(0, topk * 2)

    for tile_i in tl.static_range(0, K_TILES_PER_LAUNCH):
        offs_n = tile_start + tile_i * topk + offs_k
        valid_n = (offs_n < seq_len_kv) & (offs_n >= seq_start) & (
            offs_n < seq_end
        )

        logits = tl.zeros((topk,), dtype=tl.float32)
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)

        for h0 in tl.range(0, num_heads, BLOCK_H):
            heads = h0 + tl.arange(0, BLOCK_H)
            valid_h = heads < num_heads
            scores = tl.zeros((BLOCK_H, topk), dtype=tl.float32)
            for d0 in tl.range(0, head_dim, BLOCK_D):
                d = d0 + offs_d
                q = tl.load(
                    q_ptr
                    + row * stride_qm
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0.0,
                ).to(tl.float32)
                k = tl.load(
                    k_ptr + offs_n[None, :] * stride_kn + d[:, None] * stride_kd,
                    mask=valid_n[None, :] & (d[:, None] < head_dim),
                    other=0.0,
                ).to(tl.float32)
                scores += tl.dot(q, k, input_precision="tf32")

            weighted = tl.maximum(scores * scale[None, :], 0.0)
            weight = tl.load(
                weights_ptr + row * stride_wm + heads * stride_wh,
                mask=valid_row & valid_h,
                other=0.0,
            )
            logits += tl.sum(weighted * weight[:, None], axis=0)

        logits = tl.where(valid_row & valid_n, logits, -float("inf"))

        prev_values = tl.load(
            best_values_ptr + row * stride_bm + offs_k * stride_bk,
            mask=valid_row,
            other=-float("inf"),
        )
        prev_indices = tl.load(
            best_indices_ptr + row * stride_bm + offs_k * stride_bk,
            mask=valid_row,
            other=-1,
        )
        prev_valid = valid_row & (prev_indices >= 0)

        prev_bits = prev_values.to(tl.int32, bitcast=True)
        prev_sign = prev_bits >> 31
        prev_key = tl.where(prev_sign == 0, prev_bits ^ -1, prev_bits ^ min_i32)
        prev_key = tl.where(prev_valid, prev_key, invalid_key)
        prev_packed = ((prev_key.to(tl.int64) & 0xFFFFFFFF) << 32) | (
            prev_indices.to(tl.int64) & 0xFFFFFFFF
        )

        tile_bits = logits.to(tl.int32, bitcast=True)
        tile_sign = tile_bits >> 31
        tile_key = tl.where(tile_sign == 0, tile_bits ^ -1, tile_bits ^ min_i32)
        tile_key = tl.where(valid_row & valid_n, tile_key, invalid_key)
        tile_packed = ((tile_key.to(tl.int64) & 0xFFFFFFFF) << 32) | (
            offs_n.to(tl.int64) & 0xFFFFFFFF
        )

        merged = tl.interleave(prev_packed, tile_packed)
        sorted_merged = tl.sort(merged, descending=False)

        sorted_key = ((sorted_merged >> 32) & 0xFFFFFFFF).to(tl.int32)
        sorted_index = (sorted_merged & 0xFFFFFFFF).to(tl.int32)
        sorted_valid = sorted_key != invalid_key

        sorted_sign = sorted_key >> 31
        sorted_bits = tl.where(sorted_sign < 0, sorted_key ^ -1, sorted_key ^ min_i32)
        sorted_values = sorted_bits.to(tl.float32, bitcast=True)
        sorted_values = tl.where(sorted_valid, sorted_values, -float("inf"))
        sorted_index = tl.where(sorted_valid, sorted_index, -1)

        store_mask = valid_row & (offs_merged < topk)
        tl.store(
            best_values_ptr + row * stride_bm + offs_merged * stride_bk,
            sorted_values,
            mask=store_mask,
        )
        tl.store(
            best_indices_ptr + row * stride_bm + offs_merged * stride_bk,
            sorted_index,
            mask=store_mask,
        )


def fp8_mqa_topk_indices_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Write exact row-wise MQA top-k indices with a streaming Triton kernel.

    This is an experimental SM12x prefill-indexer path. It avoids materializing
    a full logits matrix and avoids ``torch.topk`` by keeping a persistent
    per-row top-k set and merging one ``topk``-wide KV tile at a time.
    """
    if not isinstance(kv, tuple) or len(kv) != 2:
        return False
    k_fp8, scale = kv
    tensors = (q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke, out)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not q.is_cuda or any(tensor.device != q.device for tensor in tensors):
        return False
    if (
        q.dim() != 3
        or k_fp8.dim() != 2
        or scale.dim() != 1
        or weights.dim() != 2
        or cu_seqlen_ks.dim() != 1
        or cu_seqlen_ke.dim() != 1
        or out.dim() != 2
        or q.dtype != torch.float8_e4m3fn
        or k_fp8.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or weights.dtype != torch.float32
        or cu_seqlen_ks.dtype != torch.int32
        or cu_seqlen_ke.dtype != torch.int32
        or out.dtype != torch.int32
    ):
        return False

    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    topk = out.shape[1]
    if (
        num_q == 0
        or seq_len_kv == 0
        or k_fp8.shape[1] != head_dim
        or scale.shape != (seq_len_kv,)
        or weights.shape != (num_q, num_heads)
        or cu_seqlen_ks.shape != (num_q,)
        or cu_seqlen_ke.shape != (num_q,)
        or out.shape[0] != num_q
        or topk <= 0
        or topk & (topk - 1)
        or topk not in (512, 2048)
        or (topk == 2048 and num_q < 128)
        or head_dim % 32 != 0
        or num_heads % 4 != 0
        or not q.is_contiguous()
        or not k_fp8.is_contiguous()
        or not scale.is_contiguous()
        or not weights.is_contiguous()
        or not cu_seqlen_ks.is_contiguous()
        or not cu_seqlen_ke.is_contiguous()
        or not out.is_contiguous()
    ):
        return False

    best_values = torch.empty(
        (num_q, topk),
        device=q.device,
        dtype=torch.float32,
    )
    best_values.fill_(float("-inf"))
    out.fill_(-1)

    block_h, block_d, num_warps = _fp8_mqa_topk_stream_config(topk)
    k_tiles_per_launch = _fp8_mqa_topk_stream_k_tiles_per_launch(seq_len_kv, topk)
    kv_tile_width = topk * k_tiles_per_launch
    for tile_start in range(0, seq_len_kv, kv_tile_width):
        _fp8_mqa_topk_stream_kernel[(num_q,)](
            q,
            k_fp8,
            scale,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            best_values,
            out,
            tile_start,
            num_q,
            seq_len_kv,
            topk,
            num_heads,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_fp8.stride(0),
            k_fp8.stride(1),
            weights.stride(0),
            weights.stride(1),
            best_values.stride(0),
            best_values.stride(1),
            BLOCK_D=block_d,
            BLOCK_H=block_h,
            K_TILES_PER_LAUNCH=k_tiles_per_launch,
            num_warps=num_warps,
        )
    return True


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows: tl.constexpr,
    logits_width: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset[None, :] * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[:, :, None] * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=context_mask[:, :, None] & (d[None, None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(context_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows: tl.constexpr,
    logits_width: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel optimised for long ``token_count``.

    Each Triton program handles one logical row (``batch * next_n + q_pos``)
    across a ``BLOCK_N``-wide window of token positions. Q is loaded once per
    head tile and reused for every K element in the window, which preserves
    L2 / register locality and avoids the M-axis padding waste of the
    generic 2D-tiled kernel at long contexts (mt-bench c=1 MTP=2 num_rows=3
    with token_count=131072 launches 12k programs of 128 logits each rather
    than 8k programs of 64 logits with 25 % M-axis waste).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    )

    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_pos * stride_qn
                + heads[:, None] * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[None, :] * stride_kvb
                + block_offset[None, :] * stride_kvs
                + d[:, None] * stride_kvd,
                mask=context_mask[None, :] & (d[:, None] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    """Rowwise paged-MQA logits wrapper.

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` inside ``_fp8_paged_mqa_logits_rowwise_kernel`` lands on
    tensor-core friendly tile shapes. DSv4-Flash (head_dim=128,
    num_heads=64) satisfies both and is the only model that exercises this
    path today; the generic 2D kernel below remains the fallback for
    misaligned shapes.
    """
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = torch.empty(
        (num_rows, token_count),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # Aligned head shapes (DSv4-Flash and any future MQA model with
    # ``head_dim % 64 == 0`` and ``num_heads % 4 == 0``) get the rowwise
    # kernel, which keeps long-context decode (>100K tokens) on a per-row
    # grid that re-uses Q across the full token window. The generic 2D
    # kernel below still handles misaligned shapes and remains the canonical
    # reference for the rowwise variant.
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = torch.empty(
        (num_rows, token_count),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    # Adaptive BLOCK_M: the kernel masks off positions >= num_rows, so a fixed
    # BLOCK_M=4 wastes ~75% of M-axis work in the common single-stream decode
    # case (num_rows=1). Pick the smallest power-of-2 tile that still covers
    # num_rows so we keep one grid-program for typical decode while still
    # benefiting from larger tiles when batch / MTP push num_rows higher.
    if num_rows <= 1:
        block_m = 1
    elif num_rows <= 2:
        block_m = 2
    elif num_rows <= 4:
        block_m = 4
    else:
        block_m = 8
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(token_count, 64))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


@triton.jit(do_not_specialize=["M"])
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs: tl.constexpr,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
