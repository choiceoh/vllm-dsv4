# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import os

import torch

import vllm.envs as envs
import vllm.platforms as platforms

_TRITON_SPARSE_MLA_PREFILL_TOPK_CHUNK_MIN_TOKENS = 8192


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _is_sm12x_device(device: torch.device) -> bool:
    current_platform = platforms.current_platform
    if not current_platform.is_cuda():
        return False
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    capability = current_platform.get_device_capability(device_id=index)
    return capability is not None and capability[0] == 12


def triton_sparse_mla_configured() -> bool | None:
    return envs.VLLM_TRITON_MLA_SPARSE


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return platforms.current_platform.is_device_capability_family(120)


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return _is_sm12x_device(device)


def triton_sparse_mla_topk_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_prefill_topk_chunk_size(
    num_query_tokens: int,
    num_candidates: int,
) -> int:
    base_chunk_size = max(1, triton_sparse_mla_topk_chunk_size())
    prefill_chunk_size = _env_int(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE",
        0,
    )
    if prefill_chunk_size <= 0:
        return base_chunk_size

    min_query_tokens = _env_int(
        "VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_MIN_TOKENS",
        _TRITON_SPARSE_MLA_PREFILL_TOPK_CHUNK_MIN_TOKENS,
    )
    if num_query_tokens < min_query_tokens:
        return base_chunk_size

    return min(num_candidates, max(base_chunk_size, prefill_chunk_size))


def triton_sparse_mla_query_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE


def triton_sparse_mla_head_block_size() -> int | None:
    value = envs.VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE
    if value in (1, 2, 4):
        return value
    return None


def triton_sparse_mla_prefill_blocked_accum_enabled() -> bool:
    return envs.VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM


def triton_sparse_mla_prefill_block_c() -> int:
    value = envs.VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C
    if value in (16, 32):
        return value
    return 0


def triton_sparse_mla_prefill_block_heads() -> int:
    value = envs.VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS
    if value in (1, 2, 4, 8):
        return value
    return 0


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = envs.VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE
    if configured is not None:
        return configured
    return platforms.current_platform.is_device_capability_family(120)


def triton_sparse_mla_splitkv_decode_enabled() -> bool:
    return envs.VLLM_TRITON_MLA_SPARSE_SPLITKV_DECODE
