# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional DeepSeek V4 integration hooks for the external b12x package."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

import vllm.envs as envs
import vllm.platforms as platforms
from vllm.logger import init_logger

logger = init_logger(__name__)

_INDEX_PAGE_SIZE = 64
_INDEX_HEAD_BYTES = 128
_INDEX_SCALE_BYTES = 4
_INDEX_TOKEN_BYTES = _INDEX_HEAD_BYTES + _INDEX_SCALE_BYTES
_COMPRESSED_MLA_PAYLOAD_BYTES = 576
_COMPRESSED_MLA_SCALE_BYTES = 8
_COMPRESSED_MLA_TOKEN_BYTES = (
    _COMPRESSED_MLA_PAYLOAD_BYTES + _COMPRESSED_MLA_SCALE_BYTES
)
_MHC_DEFAULT_SPLIT_K = 64
_MHC_DEFAULT_BLOCK_K = 64
_MHC_DEFAULT_BLOCK_H = 512
_PAGED_MQA_INDEX_SUPERTILE_K_ENV = "B12X_PAGED_MQA_INDEX_SUPERTILE_K"
_PAGED_MQA_INDEX_SUPERTILE_K_DEFAULT = 32768
_PAGED_MQA_INDEX_TILE_BLOCK_K = 512
_DEFAULT_ENABLED_BY_PATH = {
    # mHC is the only rs-6 hook that currently improves the 32k GB10 recipe.
    # Compressed MLA and paged-MQA need explicit opt-in while they are slower
    # than the existing vLLM paths on the measured DeepSeek V4 Flash route.
    "MHC": True,
    "COMPRESSED_MLA": False,
    "INDEXER": False,
}


def _enabled(name: str) -> bool:
    value = getattr(envs, f"VLLM_USE_B12X_DEEPSEEK_V4_{name}")
    if not envs.VLLM_USE_B12X_DEEPSEEK_V4:
        return False
    if value is None:
        return _DEFAULT_ENABLED_BY_PATH.get(name, False)
    return bool(value)


def _is_sm12x_cuda() -> bool:
    current_platform = platforms.current_platform
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    )


def _current_workspace_manager():
    from vllm.v1.worker.workspace import current_workspace_manager

    return current_workspace_manager()


@lru_cache(maxsize=1)
def _b12x_import_error() -> str | None:
    try:
        import b12x.integration  # noqa: F401
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"
    else:
        return None

    candidates: list[Path] = []
    if os.getenv("VLLM_B12X_PATH"):
        candidates.append(Path(os.environ["VLLM_B12X_PATH"]).expanduser())
    # Local development layout: /workspace/{vllm,b12x}.
    candidates.append(Path(__file__).resolve().parents[6] / "b12x")

    for candidate in candidates:
        if not (candidate / "b12x" / "__init__.py").exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            import b12x.integration  # noqa: F401
        except Exception as exc:
            first_error = f"{type(exc).__name__}: {exc}"
            continue
        return None
    return first_error


def _b12x_available(path_name: str) -> bool:
    error = _b12x_import_error()
    if error is None:
        return True
    logger.warning_once("B12x %s path disabled: %s", path_name, error)
    return False


def _shape(cache: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in cache.shape)


def _skip_once(path_name: str, reason: str) -> None:
    logger.info_once("B12x %s path skipped: %s", path_name, reason)


def _compressed_mla_page_nbytes(page_size: int) -> int:
    unpadded = int(page_size) * _COMPRESSED_MLA_TOKEN_BYTES
    return (
        (unpadded + _COMPRESSED_MLA_PAYLOAD_BYTES - 1)
        // _COMPRESSED_MLA_PAYLOAD_BYTES
        * _COMPRESSED_MLA_PAYLOAD_BYTES
    )


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _native_page_view_from_rank3(
    cache: torch.Tensor,
    *,
    expected_width: int,
    require_contiguous: bool,
    path_name: str,
) -> torch.Tensor | None:
    page_stride = int(cache.stride(0))
    if int(cache.stride(-1)) != 1:
        _skip_once(
            path_name,
            "rank-3 cache last dimension is not contiguous "
            f"shape={_shape(cache)} stride={tuple(int(s) for s in cache.stride())}",
        )
        return None
    if page_stride < int(expected_width):
        _skip_once(
            path_name,
            "rank-3 cache page stride is too small "
            f"shape={_shape(cache)} stride={tuple(int(s) for s in cache.stride())} "
            f"expected_width={int(expected_width)}",
        )
        return None

    pages = int(cache.shape[0])
    storage_required = int(cache.storage_offset())
    if pages > 0:
        storage_required += (pages - 1) * page_stride + int(expected_width)
    storage_size = _storage_nbytes(cache) // int(cache.element_size())
    if storage_required > storage_size:
        _skip_once(
            path_name,
            "rank-3 cache storage is too small for native page view "
            f"shape={_shape(cache)} stride={tuple(int(s) for s in cache.stride())} "
            f"expected_width={int(expected_width)} storage_required={storage_required} "
            f"storage_size={storage_size}",
        )
        return None

    view = torch.as_strided(
        cache,
        size=(pages, int(expected_width)),
        stride=(page_stride, 1),
    )
    if require_contiguous and not view.is_contiguous():
        _skip_once(
            path_name,
            "native page view is not contiguous "
            f"shape={_shape(view)} stride={tuple(int(s) for s in view.stride())}",
        )
        return None
    return view


def _compressed_cache_view(
    cache: torch.Tensor,
    *,
    page_size: int,
    path_name: str,
) -> torch.Tensor | None:
    if cache.dtype != torch.uint8:
        _skip_once(path_name, f"cache dtype is {cache.dtype}, expected torch.uint8")
        return None

    expected_page_nbytes = _compressed_mla_page_nbytes(page_size)
    if cache.ndim == 2:
        if int(cache.shape[1]) != expected_page_nbytes:
            _skip_once(
                path_name,
                "native page width mismatch "
                f"shape={_shape(cache)} expected_width={expected_page_nbytes}",
            )
            return None
        if not cache.is_contiguous():
            _skip_once(path_name, "native page cache is not contiguous")
            return None
        return cache

    if (
        cache.ndim == 3
        and int(cache.shape[1]) == int(page_size)
        and int(cache.shape[2]) == _COMPRESSED_MLA_TOKEN_BYTES
    ):
        view = _native_page_view_from_rank3(
            cache,
            expected_width=expected_page_nbytes,
            require_contiguous=True,
            path_name=path_name,
        )
        if view is not None:
            logger.info_once(
                "Using B12x native compressed MLA page view for %s cache "
                "shape=%s stride=%s native_shape=%s native_stride=%s.",
                path_name,
                _shape(cache),
                tuple(int(s) for s in cache.stride()),
                _shape(view),
                tuple(int(s) for s in view.stride()),
            )
        return view

    _skip_once(
        path_name,
        "unsupported compressed cache layout "
        f"shape={_shape(cache)} page_size={int(page_size)}",
    )
    return None


def _index_cache_view(cache: torch.Tensor, *, path_name: str) -> torch.Tensor | None:
    expected_width = _INDEX_PAGE_SIZE * _INDEX_TOKEN_BYTES
    if cache.dtype != torch.uint8:
        _skip_once(path_name, f"cache dtype is {cache.dtype}, expected torch.uint8")
        return None

    if cache.ndim == 2:
        if int(cache.shape[1]) != expected_width:
            _skip_once(
                path_name,
                "native page width mismatch "
                f"shape={_shape(cache)} expected_width={expected_width}",
            )
            return None
        if int(cache.stride(-1)) != 1:
            _skip_once(path_name, "native page cache last dimension is not contiguous")
            return None
        return cache

    if (
        cache.ndim == 3
        and int(cache.shape[1]) == _INDEX_PAGE_SIZE
        and int(cache.shape[2]) == _INDEX_TOKEN_BYTES
    ):
        view = _native_page_view_from_rank3(
            cache,
            expected_width=expected_width,
            require_contiguous=False,
            path_name=path_name,
        )
        if view is not None:
            logger.info_once(
                "Using B12x native paged MQA index cache view "
                "shape=%s stride=%s native_shape=%s native_stride=%s.",
                _shape(cache),
                tuple(int(s) for s in cache.stride()),
                _shape(view),
                tuple(int(s) for s in view.stride()),
            )
        return view

    _skip_once(path_name, f"unsupported index cache layout shape={_shape(cache)}")
    return None


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _paged_mqa_supertile_tokens(*, page_width: int | None = None) -> int:
    raw = os.environ.get(_PAGED_MQA_INDEX_SUPERTILE_K_ENV)
    if raw is None:
        supertile_k = _PAGED_MQA_INDEX_SUPERTILE_K_DEFAULT
    else:
        try:
            supertile_k = int(raw)
        except ValueError:
            supertile_k = _PAGED_MQA_INDEX_SUPERTILE_K_DEFAULT
    supertile_k = _align_up(
        max(int(supertile_k), _PAGED_MQA_INDEX_TILE_BLOCK_K),
        _PAGED_MQA_INDEX_TILE_BLOCK_K,
    )
    if page_width is not None:
        live_tokens = max(int(page_width), 1) * _INDEX_PAGE_SIZE
        live_supertile_k = _align_up(
            max(live_tokens, _PAGED_MQA_INDEX_TILE_BLOCK_K),
            _PAGED_MQA_INDEX_TILE_BLOCK_K,
        )
        supertile_k = min(supertile_k, live_supertile_k)
    return max(supertile_k, _PAGED_MQA_INDEX_TILE_BLOCK_K)


_compressed_mla_workspaces: dict[tuple[int, int, int, int, int], object] = {}
_paged_indexer_workspaces: dict[tuple[int, int, int, int, int], object] = {}


def _workspace_key(
    device: torch.device,
    *,
    rows: int,
    heads: int,
    topk: int,
    page_width: int,
) -> tuple[int, int, int, int, int]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return (int(index), int(rows), int(heads), int(topk), int(page_width))


def _get_compressed_mla_workspace(
    *,
    device: torch.device,
    rows: int,
    heads: int,
    topk: int,
) -> object | None:
    key = _workspace_key(
        device, rows=rows, heads=heads, topk=topk, page_width=topk
    )
    workspace = _compressed_mla_workspaces.get(key)
    if workspace is not None:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        return None
    from b12x.integration import B12XAttentionWorkspace

    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=heads,
        head_dim=512,
        v_head_dim=512,
        topk=topk,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=0,
        use_cuda_graph=True,
        reserve_compressed_mla_metadata=True,
    )
    _compressed_mla_workspaces[key] = workspace
    return workspace


def _get_paged_indexer_workspace(
    *,
    device: torch.device,
    rows: int,
    heads: int,
    topk: int,
    page_width: int,
    supertile_k: int,
) -> object | None:
    supertile_pages = max(1, int(supertile_k) // _INDEX_PAGE_SIZE)
    page_width = max(int(page_width), supertile_pages)
    key = _workspace_key(
        device, rows=rows, heads=heads, topk=topk, page_width=page_width
    )
    workspace = _paged_indexer_workspaces.get(key)
    if workspace is not None:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        return None
    from b12x.integration import B12XAttentionWorkspace

    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=heads,
        indexer_num_q_heads=heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=page_width,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=_INDEX_PAGE_SIZE,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=supertile_k,
    )
    _paged_indexer_workspaces[key] = workspace
    return workspace


def b12x_compressed_mla_decode(
    *,
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    compressed_page_size: int | None,
    swa_page_size: int,
    sm_scale: float,
    attn_sink: torch.Tensor,
    num_heads: int,
    output: torch.Tensor,
) -> bool:
    if not (_enabled("COMPRESSED_MLA") and _is_sm12x_cuda()):
        return False
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512:
        _skip_once("compressed MLA", f"q shape is {_shape(q)}, expected [T, 1, H, 512]")
        return False
    if int(q.shape[2]) < int(num_heads):
        _skip_once(
            "compressed MLA",
            f"q has {int(q.shape[2])} heads, expected at least {int(num_heads)}",
        )
        return False
    rows = int(q.shape[0])
    if rows == 0:
        return True
    swa_cache_2d = _compressed_cache_view(
        swa_k_cache, page_size=swa_page_size, path_name="compressed MLA SWA"
    )
    if swa_cache_2d is None:
        return False

    indexed_cache_2d = None
    indexed_indices_2d = None
    indexed_lens_1d = None
    if compressed_k_cache is not None:
        if compressed_page_size is None or topk_indices is None or topk_lens is None:
            _skip_once("compressed MLA", "indexed cache metadata is incomplete")
            return False
        indexed_cache_2d = _compressed_cache_view(
            compressed_k_cache,
            page_size=compressed_page_size,
            path_name="compressed MLA indexed",
        )
        if indexed_cache_2d is None:
            return False
        indexed_indices_2d = topk_indices[:, 0, :] if topk_indices.ndim == 3 else topk_indices
        indexed_lens_1d = topk_lens

    q_src = q[:, 0, :num_heads, :]
    if q_src.dtype != torch.bfloat16:
        _skip_once("compressed MLA", f"q dtype is {q_src.dtype}, expected torch.bfloat16")
        return False
    if q_src.is_contiguous():
        q_live = q_src
    else:
        try:
            (q_live,) = _current_workspace_manager().get_simultaneous(
                ((rows, num_heads, q.shape[-1]), q.dtype),
            )
        except AssertionError:
            _skip_once("compressed MLA", "workspace allocation was unavailable")
            return False
        q_live.copy_(q_src)

    swa_indices_2d = swa_indices[:, 0, :] if swa_indices.ndim == 3 else swa_indices
    if swa_indices_2d.shape[0] != rows or swa_lens.shape[0] < rows:
        _skip_once(
            "compressed MLA",
            "SWA metadata row mismatch "
            f"indices={_shape(swa_indices_2d)} lens={_shape(swa_lens)} rows={rows}",
        )
        return False

    if not _b12x_available("compressed MLA"):
        return False

    live_topk = int(swa_indices_2d.shape[1]) + (
        int(indexed_indices_2d.shape[1]) if indexed_indices_2d is not None else 0
    )
    workspace = _get_compressed_mla_workspace(
        device=q.device,
        rows=rows,
        heads=num_heads,
        topk=max(live_topk, 1),
    )
    if workspace is None:
        _skip_once("compressed MLA", "workspace creation is unavailable during capture")
        return False

    try:
        from b12x.integration import compressed_mla_decode_forward

        result = compressed_mla_decode_forward(
            q_all=q_live,
            swa_k_cache=swa_cache_2d,
            swa_indices=swa_indices_2d,
            swa_topk_lengths=swa_lens[:rows],
            workspace=workspace,
            sm_scale=sm_scale,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_cache_2d,
            indexed_indices=indexed_indices_2d,
            indexed_topk_lengths=indexed_lens_1d,
            indexed_page_size=compressed_page_size,
            attn_sink=attn_sink[:num_heads],
            expected_num_q_heads=num_heads,
        )
    except Exception as exc:
        logger.warning_once(
            "B12x compressed MLA path failed; falling back to vLLM kernels: %s",
            exc,
        )
        return False

    output[:, :num_heads, :].copy_(result)
    if output.shape[1] > num_heads:
        output[:, num_heads:, :].zero_()
    logger.info_once("Using B12x rs-6 compressed MLA decode path.")
    return True


def _flatten_page_metadata(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if seq_lens.ndim == 2:
        batch, next_n = map(int, seq_lens.shape)
        if block_table.shape[0] < batch:
            return None
        if next_n == 1:
            return block_table[:batch].contiguous(), seq_lens.reshape(-1).contiguous()
        return (
            block_table[:batch].repeat_interleave(next_n, dim=0).contiguous(),
            seq_lens.reshape(-1).contiguous(),
        )
    if seq_lens.ndim == 1:
        rows = int(seq_lens.shape[0])
        if block_table.shape[0] < rows:
            return None
        return block_table[:rows].contiguous(), seq_lens.contiguous()
    return None


def b12x_paged_mqa_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> bool:
    if not (_enabled("INDEXER") and _is_sm12x_cuda()):
        return False
    if q_fp8.ndim != 4 or q_fp8.shape[-1] != _INDEX_HEAD_BYTES:
        _skip_once(
            "paged MQA indexer",
            f"q shape is {_shape(q_fp8)}, expected [B, N, H, {_INDEX_HEAD_BYTES}]",
        )
        return False
    if q_fp8.dtype != torch.float8_e4m3fn or weights.dtype != torch.float32:
        _skip_once(
            "paged MQA indexer",
            f"q/weights dtypes are {q_fp8.dtype}/{weights.dtype}, expected "
            "torch.float8_e4m3fn/torch.float32",
        )
        return False
    index_cache_2d = _index_cache_view(
        index_k_cache, path_name="paged MQA indexer"
    )
    if index_cache_2d is None:
        return False

    flat_metadata = _flatten_page_metadata(
        block_table=block_table,
        seq_lens=seq_lens,
    )
    if flat_metadata is None:
        _skip_once(
            "paged MQA indexer",
            f"metadata layout unsupported seq_lens={_shape(seq_lens)} "
            f"block_table={_shape(block_table)}",
        )
        return False
    real_page_table, cache_seqlens = flat_metadata

    rows = int(q_fp8.shape[0]) * int(q_fp8.shape[1])
    heads = int(q_fp8.shape[2])
    q_flat = q_fp8.reshape(rows, heads, _INDEX_HEAD_BYTES)
    if weights.shape[0] < rows or weights.shape[1] != heads:
        _skip_once(
            "paged MQA indexer",
            f"weights shape is {_shape(weights)}, expected at least [{rows}, {heads}]",
        )
        return False
    weights_flat = weights[:rows]
    if real_page_table.shape[0] != rows or cache_seqlens.shape[0] != rows:
        _skip_once(
            "paged MQA indexer",
            "flattened metadata row mismatch "
            f"page_table={_shape(real_page_table)} seqlens={_shape(cache_seqlens)} "
            f"rows={rows}",
        )
        return False
    if topk_indices.shape[0] < rows or topk_indices.shape[1] < topk:
        _skip_once(
            "paged MQA indexer",
            f"topk buffer shape is {_shape(topk_indices)}, expected at least "
            f"[{rows}, {topk}]",
        )
        return False

    if not _b12x_available("paged MQA indexer"):
        return False

    page_width = int(real_page_table.shape[1])
    supertile_k = _paged_mqa_supertile_tokens(page_width=page_width)
    workspace = _get_paged_indexer_workspace(
        device=q_fp8.device,
        rows=rows,
        heads=heads,
        topk=topk,
        page_width=page_width,
        supertile_k=supertile_k,
    )
    if workspace is None:
        _skip_once("paged MQA indexer", "workspace creation is unavailable during capture")
        return False

    try:
        from b12x.integration import (
            paged_mqa_index_decode_supertile_topk_fp8,
            prepare_paged_mqa_indexer_metadata,
        )

        metadata = prepare_paged_mqa_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens,
            expected_num_q_heads=heads,
            build_schedule=False,
            validate_raw_lengths=False,
        )
        paged_mqa_index_decode_supertile_topk_fp8(
            q_fp8=q_flat,
            weights=weights_flat,
            index_k_cache=index_cache_2d,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=heads,
            workspace=workspace,
            out_indices=topk_indices[:rows, :topk],
            supertile_k=supertile_k,
        )
    except Exception as exc:
        logger.warning_once(
            "B12x paged MQA indexer path failed; falling back to vLLM kernels: %s",
            exc,
        )
        return False
    logger.info_once("Using B12x rs-6 paged MQA indexer decode path.")
    return True


def b12x_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not (_enabled("MHC") and _is_sm12x_cuda()):
        return None
    if not _b12x_available("mHC"):
        return None
    if (
        hc_pre_eps != hc_sinkhorn_eps
        or hc_post_mult_value != 2.0
        or residual.dtype != torch.bfloat16
        or residual.shape[-2] != 4
        or not residual.is_contiguous()
    ):
        return None

    outer_shape = residual.shape[:-2]
    tokens = 1
    for dim in outer_shape:
        tokens *= int(dim)
    hidden_size = int(residual.shape[-1])
    residual_flat = residual.view(tokens, 4, hidden_size)
    try:
        (
            partials,
            y_out,
            post_out,
            comb_out,
        ) = _current_workspace_manager().get_simultaneous(
            ((tokens, _MHC_DEFAULT_SPLIT_K, 25), torch.float32),
            ((tokens, hidden_size), torch.bfloat16),
            ((tokens, 4), torch.float32),
            ((tokens, 4, 4), torch.float32),
        )
    except AssertionError:
        return None
    try:
        from b12x.integration import b12x_mhc_pre as _b12x_mhc_pre

        y, post, comb = _b12x_mhc_pre(
            residual_flat,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_pre_eps,
            sinkhorn_iters=sinkhorn_repeat,
            workspace=partials,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            split_k=_MHC_DEFAULT_SPLIT_K,
            block_k=_MHC_DEFAULT_BLOCK_K,
            block_h=_MHC_DEFAULT_BLOCK_H,
        )
    except Exception as exc:
        logger.warning_once(
            "B12x mHC pre path failed; falling back to vLLM kernels: %s",
            exc,
        )
        return None
    logger.info_once("Using B12x rs-6 mHC pre path.")
    return (
        post.view(*outer_shape, 4, 1),
        comb.view(*outer_shape, 4, 4),
        y.view(*outer_shape, hidden_size),
    )


def b12x_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor | None:
    if not (_enabled("MHC") and _is_sm12x_cuda()):
        return None
    if not _b12x_available("mHC"):
        return None
    if (
        x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or residual.shape[-2] != 4
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or not post.is_contiguous()
        or not comb.is_contiguous()
    ):
        return None

    outer_shape = residual.shape[:-2]
    tokens = 1
    for dim in outer_shape:
        tokens *= int(dim)
    hidden_size = int(residual.shape[-1])
    x_flat = x.view(tokens, hidden_size)
    residual_flat = residual.view(tokens, 4, hidden_size)
    post_flat = post.view(tokens, 4) if post.shape[-1] == 1 else post.view(tokens, 4)
    comb_flat = comb.view(tokens, 4, 4)
    try:
        (out,) = _current_workspace_manager().get_simultaneous(
            ((tokens, 4, hidden_size), torch.bfloat16),
        )
    except AssertionError:
        return None
    try:
        from b12x.integration import b12x_mhc_post as _b12x_mhc_post

        result = _b12x_mhc_post(
            x_flat,
            residual_flat,
            post_flat,
            comb_flat,
            out=out,
            block_h=_MHC_DEFAULT_BLOCK_H,
        )
    except Exception as exc:
        logger.warning_once(
            "B12x mHC post path failed; falling back to vLLM kernels: %s",
            exc,
        )
        return None
    logger.info_once("Using B12x rs-6 mHC post path.")
    return result.view(*outer_shape, 4, hidden_size)
