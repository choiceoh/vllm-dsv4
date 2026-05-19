# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x fallback implementations for DeepGEMM-only interfaces."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 * 1024 * 1024
_SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES = 512 * 1024 * 1024
_SM120_PAGED_MQA_TOPK_CHUNK_SIZE = 8192
_SM120_MQA_TOPK_TRITON_MIN_ROWS = 64
_SM120_MQA_TOPK_TRITON_MIN_ROWS_TOPK2048 = 128
_SM120_MQA_TOPK_TRITON_MAX_ROWS = 256
_SM120_MQA_TOPK_TRITON_MIN_KV_TOKENS = 8192


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer value for %s=%r", name, value)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _top_k_per_row_prefill_op():
    try:
        from vllm import _custom_ops as _custom_ops  # noqa: F401

        return torch.ops._C.top_k_per_row_prefill
    except (AttributeError, ImportError, RuntimeError):
        return None


def _use_triton_prefill_mqa_topk(
    q_values: torch.Tensor,
    k_values: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    tile_size = _triton_prefill_mqa_topk_row_tile_size(
        q_values,
        k_values,
        topk_indices,
    )
    return tile_size > 0 and q_values.shape[0] <= tile_size


def _triton_prefill_mqa_topk_row_tile_size(
    q_values: torch.Tensor,
    k_values: torch.Tensor,
    topk_indices: torch.Tensor,
) -> int:
    if os.getenv("VLLM_SM12X_MQA_TOPK_TRITON", "0") != "1":
        return 0
    min_rows = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS",
        _SM120_MQA_TOPK_TRITON_MIN_ROWS,
    )
    min_kv_tokens = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS",
        _SM120_MQA_TOPK_TRITON_MIN_KV_TOKENS,
    )
    max_rows = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS",
        _SM120_MQA_TOPK_TRITON_MAX_ROWS,
    )
    row_count = q_values.shape[0]
    topk_tokens = topk_indices.shape[1]
    min_rows_for_topk = (
        max(min_rows, _SM120_MQA_TOPK_TRITON_MIN_ROWS_TOPK2048)
        if topk_tokens == 2048
        else min_rows
    )
    if (
        row_count < min_rows_for_topk
        or k_values.shape[0] < min_kv_tokens
        or topk_tokens not in (512, 2048)
    ):
        return 0
    if max_rows <= 0:
        return row_count
    if max_rows < min_rows_for_topk:
        return 0
    return max_rows


def _iter_mqa_topk_row_tiles(
    row_count: int,
    tile_size: int,
    min_rows: int,
):
    row_start = 0
    while row_start < row_count:
        remaining = row_count - row_start
        if remaining <= tile_size:
            yield row_start, row_count
            return

        row_end = row_start + tile_size
        tail = row_count - row_end
        if 0 < tail < min_rows:
            row_end -= min_rows - tail
        yield row_start, row_end
        row_start = row_end


def _fp8_mqa_logits_head_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
) -> int:
    # The SM120 torch path is used on long prefill paths where materializing
    # [head_chunk, M, N] scores can otherwise allocate multiple GiB. Keep the
    # transient score tensor bounded, while still using larger head chunks for
    # short prompts where they are faster.
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_head * 4)
    return max(1, min(8, num_heads, max_heads))


def _fp8_mqa_logits_k_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    head_chunk_size: int,
) -> int:
    score_elems_per_key = max(1, seq_len * head_chunk_size)
    max_keys = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_key * 4)
    return max(1, min(seq_len_kv, max_keys))


def _fp8_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)

    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
        q_chunk = q_chunk.transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        k_chunk_size = _fp8_mqa_logits_k_chunk_size(
            seq_len, seq_len_kv, head_end - head_start
        )
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(
                scores[0] if scores.shape[0] == 1 else scores.sum(dim=0)
            )

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def _fp8_mqa_logits_topk_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA top-k torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    if out is None:
        out = torch.empty(
            (seq_len, topk_tokens), device=q_values.device, dtype=torch.int32
        )
    else:
        assert out.shape == (seq_len, topk_tokens)
        assert out.dtype == torch.int32
    out.fill_(-1)

    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)
    k_chunk_size = _fp8_mqa_logits_k_chunk_size(seq_len, seq_len_kv, head_chunk_size)
    max_chunk_topk = min(topk_tokens, k_chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, k_chunk_size):
        k_end = min(k_start + k_chunk_size, seq_len_kv)
        chunk_logits = torch.zeros(
            (seq_len, k_end - k_start),
            device=q_values.device,
            dtype=torch.float32,
        )
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
            q_chunk = q_chunk.transpose(0, 1).contiguous()
            head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            chunk_logits.add_(scores[0] if scores.shape[0] == 1 else scores.sum(dim=0))

        offsets = torch.arange(k_start, k_end, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        chunk_logits.masked_fill_(~valid, float("-inf"))

        chunk_topk = min(topk_tokens, k_end - k_start)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return out


def _fp8_mqa_logits_topk_triton(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    k_values, _ = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    logits_bytes = q_values.shape[0] * k_values.shape[0] * torch.float32.itemsize
    if logits_bytes > _SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES:
        return False

    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    logits = fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    topk_tokens = out.shape[1]
    select_k = min(topk_tokens, logits.shape[1])
    out.fill_(-1)
    if select_k == 0:
        return True

    selected = out[:, :select_k]
    topk_op = _top_k_per_row_prefill_op()
    if topk_op is not None:
        topk_op(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            selected,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            select_k,
        )
        selected.add_(cu_seqlen_ks[:, None])
        valid = (selected >= cu_seqlen_ks[:, None]) & (
            selected < cu_seqlen_ke[:, None]
        )
        selected.masked_fill_(~valid, -1)
    else:
        values, indices = torch.topk(logits, select_k, dim=1)
        selected.copy_(indices.to(torch.int32))
        selected.masked_fill_(~torch.isfinite(values), -1)
    return True


def _fp8_mqa_logits_topk_triton_row_tiled(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    if not _env_bool("VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED"):
        return False

    q_values, q_scale = q
    k_values, _ = kv
    if not (
        q_scale is None
        and q_values.dim() == 3
        and k_values.dim() == 2
        and out.dim() == 2
        and q_values.shape[0] == out.shape[0]
        and q_values.device == k_values.device
        and q_values.device == out.device
        and out.dtype == torch.int32
    ):
        return False

    row_count = q_values.shape[0]
    kv_tokens = k_values.shape[0]
    topk_tokens = out.shape[1]
    if row_count == 0:
        return True
    if kv_tokens == 0 or topk_tokens == 0:
        out.fill_(-1)
        return True

    configured_tile = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE",
        _env_int(
            "VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS",
            _SM120_MQA_TOPK_TRITON_MAX_ROWS,
        ),
    )
    if configured_tile <= 0:
        configured_tile = row_count

    bytes_per_row = kv_tokens * torch.float32.itemsize
    max_rows_by_bytes = max(
        1, _SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES // bytes_per_row
    )
    tile_size = min(row_count, configured_tile, max_rows_by_bytes)

    configured_min_rows = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS",
        _SM120_MQA_TOPK_TRITON_MIN_ROWS,
    )
    min_rows = (
        max(configured_min_rows, _SM120_MQA_TOPK_TRITON_MIN_ROWS_TOPK2048)
        if topk_tokens == 2048
        else configured_min_rows
    )
    min_kv_tokens = _env_int(
        "VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS",
        _SM120_MQA_TOPK_TRITON_MIN_KV_TOKENS,
    )
    if (
        tile_size < min_rows
        or kv_tokens < min_kv_tokens
        or topk_tokens not in (512, 2048)
    ):
        return False

    tile_count = 0
    for row_start, row_end in _iter_mqa_topk_row_tiles(
        row_count,
        tile_size,
        min_rows,
    ):
        q_tile = q_values[row_start:row_end]
        weights_tile = weights[row_start:row_end]
        cu_seqlen_ks_tile = cu_seqlen_ks[row_start:row_end]
        cu_seqlen_ke_tile = cu_seqlen_ke[row_start:row_end]
        out_tile = out[row_start:row_end]
        out_copy_back = out_tile if not out_tile.is_contiguous() else None
        if out_copy_back is not None:
            out_tile = torch.empty(
                (row_end - row_start, topk_tokens),
                device=out.device,
                dtype=out.dtype,
            )
        try:
            used_triton = _fp8_mqa_logits_topk_triton(
                (q_tile if q_tile.is_contiguous() else q_tile.contiguous(), None),
                kv,
                weights_tile
                if weights_tile.is_contiguous()
                else weights_tile.contiguous(),
                cu_seqlen_ks_tile
                if cu_seqlen_ks_tile.is_contiguous()
                else cu_seqlen_ks_tile.contiguous(),
                cu_seqlen_ke_tile
                if cu_seqlen_ke_tile.is_contiguous()
                else cu_seqlen_ke_tile.contiguous(),
                out_tile,
            )
        except Exception as exc:
            logger.warning_once(
                "SM12x Triton materialized MQA logits top-k row-tiled path "
                "rejected at runtime; falling back to streaming top-k "
                "(q_rows=%s, row_tile=%s, kv_tokens=%s, topk=%s, "
                "error=%s: %s).",
                row_count,
                row_end - row_start,
                kv_tokens,
                topk_tokens,
                type(exc).__name__,
                exc,
            )
            return False
        if not used_triton:
            return False
        if out_copy_back is not None:
            out_copy_back.copy_(out_tile)
        tile_count += 1

    logger.info_once(
        "Using SM12x Triton materialized MQA logits top-k row-tiled path "
        "(q_rows=%s, row_tile=%s, row_tiles=%s, kv_tokens=%s, topk=%s).",
        row_count,
        tile_size,
        tile_count,
        kv_tokens,
        topk_tokens,
    )
    return True


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    if _fp8_mqa_logits_topk_triton_row_tiled(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True

    tile_size = _triton_prefill_mqa_topk_row_tile_size(q[0], kv[0], topk_indices)
    if tile_size > 0:
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
            fp8_mqa_topk_indices_triton,
        )

        row_count = q[0].shape[0]
        topk_tokens = topk_indices.shape[1]
        configured_min_rows = _env_int(
            "VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS",
            _SM120_MQA_TOPK_TRITON_MIN_ROWS,
        )
        min_rows = (
            max(configured_min_rows, _SM120_MQA_TOPK_TRITON_MIN_ROWS_TOPK2048)
            if topk_tokens == 2048
            else configured_min_rows
        )
        tile_count = 0
        for row_start, row_end in _iter_mqa_topk_row_tiles(
            row_count,
            tile_size,
            min_rows,
        ):
            q_tile = q[0][row_start:row_end]
            weights_tile = weights[row_start:row_end]
            cu_seqlen_ks_tile = cu_seqlen_ks[row_start:row_end]
            cu_seqlen_ke_tile = cu_seqlen_ke[row_start:row_end]
            topk_tile = topk_indices[row_start:row_end]
            topk_copy_back = topk_tile if not topk_tile.is_contiguous() else None
            if topk_copy_back is not None:
                topk_tile = torch.empty(
                    (row_end - row_start, topk_tokens),
                    device=topk_indices.device,
                    dtype=topk_indices.dtype,
                )
            try:
                used_triton = fp8_mqa_topk_indices_triton(
                    q_tile if q_tile.is_contiguous() else q_tile.contiguous(),
                    kv,
                    weights_tile
                    if weights_tile.is_contiguous()
                    else weights_tile.contiguous(),
                    cu_seqlen_ks_tile
                    if cu_seqlen_ks_tile.is_contiguous()
                    else cu_seqlen_ks_tile.contiguous(),
                    cu_seqlen_ke_tile
                    if cu_seqlen_ke_tile.is_contiguous()
                    else cu_seqlen_ke_tile.contiguous(),
                    topk_tile,
                )
            except Exception as exc:
                logger.warning_once(
                    "SM12x Triton prefill MQA top-k path rejected at runtime; "
                    "falling back to torch top-k (q_rows=%s, row_tile=%s, "
                    "kv_tokens=%s, topk=%s, error=%s: %s).",
                    row_count,
                    row_end - row_start,
                    kv[0].shape[0],
                    topk_tokens,
                    type(exc).__name__,
                    exc,
                )
                break
            if not used_triton:
                break
            if topk_copy_back is not None:
                topk_copy_back.copy_(topk_tile)
            tile_count += 1
        else:
            logger.warning_once(
                "Using SM12x Triton prefill MQA top-k path "
                "(q_rows=%s, row_tile=%s, row_tiles=%s, kv_tokens=%s, "
                "topk=%s).",
                row_count,
                tile_size,
                tile_count,
                kv[0].shape[0],
                topk_tokens,
            )
            return True

    if _fp8_mqa_logits_topk_triton(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True

    _fp8_mqa_logits_topk_torch(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices.shape[1],
        out=topk_indices,
    )
    return True


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if clean_logits and q_scale is None and q_values.dim() == 3 and kv[0].dim() == 2:
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
            fp8_mqa_logits_triton,
        )

        return fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return _fp8_mqa_logits_torch(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def _fp8_paged_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA torch path only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    head_dim_with_scale = kv_cache.shape[-1]
    assert head_dim_with_scale > head_dim
    assert weights.shape == (batch_size * next_n, num_heads)
    assert context_lens.shape == (batch_size, next_n)

    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
        _view_packed_fp8_paged_mqa_kv_cache,
    )

    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    q_f32 = q_values.float()
    score_bytes = _SM120_MQA_LOGITS_MAX_SCORE_BYTES
    max_tokens_per_chunk = max(1, score_bytes // max(1, num_heads * 4))
    token_offsets_cache: dict[int, torch.Tensor] = {}

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            for token_start in range(0, context_len, max_tokens_per_chunk):
                token_end = min(context_len, token_start + max_tokens_per_chunk)
                chunk_len = token_end - token_start
                token_offsets = token_offsets_cache.get(chunk_len)
                if token_offsets is None or token_offsets.device != q_values.device:
                    token_offsets = torch.arange(
                        chunk_len, device=q_values.device, dtype=torch.long
                    )
                    token_offsets_cache[chunk_len] = token_offsets
                token_ids = token_start + token_offsets
                logical_blocks = token_ids // block_kv
                token_in_block = token_ids - logical_blocks * block_kv
                physical_blocks = block_tables[batch_idx, logical_blocks]
                kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
                scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
                kv_chunk.mul_(scale_chunk[:, None])
                scores = torch.matmul(q_row, kv_chunk.T)
                scores.relu_()
                scores.mul_(row_weights[:, None])
                logits[row, token_start:token_end] = scores.sum(dim=0)

    return logits


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if (
        q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
            fp8_paged_mqa_logits_triton,
        )

        return fp8_paged_mqa_logits_triton(
            q_values, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    logger.warning_once(
        "SM12x paged-MQA falling back to the torch reference path "
        "(q_scale=%s, q.dim=%s, kv_cache.dtype=%s, kv_cache.shape[-1]=%s, "
        "q_values.shape[-1]=%s). This path is intended for correctness checks "
        "and is not graph-compatible; expect a large per-step latency.",
        "set" if q_scale is not None else "None",
        q_values.dim(),
        kv_cache.dtype,
        kv_cache.shape[-1] if kv_cache.dim() else None,
        q_values.shape[-1],
    )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    q_values, q_scale = q
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        return False

    num_rows = q_values.shape[0] * q_values.shape[1]
    topk_tokens = topk_indices.shape[1]
    assert topk_indices.shape == (num_rows, topk_tokens)
    assert topk_indices.dtype == torch.int32
    topk_indices.fill_(-1)
    if num_rows == 0 or topk_tokens == 0 or max_model_len == 0:
        return True

    best_values = torch.full(
        (num_rows, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_size = max(1, _SM120_PAGED_MQA_TOPK_CHUNK_SIZE)
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
        fp8_paged_mqa_logits_triton,
    )

    for token_start in range(0, max_model_len, chunk_size):
        token_count = min(chunk_size, max_model_len - token_start)
        chunk_logits = fp8_paged_mqa_logits_triton(
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )
        chunk_topk = min(topk_tokens, token_count)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(token_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(topk_indices)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=topk_indices)
        best_values, next_best_values = next_best_values, best_values
        topk_indices.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def _tf32_hc_prenorm_gemm_torch(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Portable SM12x HyperConnection prenorm GEMM fallback.

    DeepGEMM's split ABI only requires that downstream consumers recover the
    full result by summing over the split dimension. Keep the implementation
    simple by writing the full product to split zero and clearing the rest.
    """
    del num_split
    product = x.float() @ fn.float().T
    norm = x.float().square().sum(dim=-1)

    if out.dim() == 3:
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(product)
        sqrsum[0].copy_(norm)
    else:
        out.copy_(product)
        sqrsum.copy_(norm)
    return out


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    if out.dim() == 3 and sqrsum.dim() == 2:
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
            tf32_hc_prenorm_gemm_triton,
        )

        tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)
        return out

    return _tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split)
