# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "V4_FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16
# Cap warmup at the largest single-chunk prefill the scheduler will ever
# issue (max_num_batched_tokens). 8192 covers the canonical SM12x serve
# (max_num_batched_tokens=8192); larger scheduler caps clamp to this
# value via _clamp_warmup_tokens at the call site, smaller caps clamp
# down naturally.
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 8192
# Steady-state MTP decode shapes to warm. Keep this bounded to the edge
# deployment range we expect to optimize; warming the scheduler's raw
# max_num_seqs can consume multiple GiB of temporary workspace on
# long-context SM12x serves before the first request.
_DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS = (1, 2, 4, 8, 16, 24, 32)
_DEEPSEEK_V4_MTP_UNIFORM_DECODE_MAX_WARMUP_REQUESTS = 32
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = tuple(range(1, 17)) + (
    32,
    64,
    128,
    256,
    512,
)
_DEEPSEEK_V4_METADATA_WARMUP_TOKENS = 16
_DEEPSEEK_V4_ROUTE_PACK_WARMUP_TOKENS = (1, 3, 512)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _is_deepseek_v4_mtp_spec_decode(runner: "GPUModelRunner") -> bool:
    spec_config = getattr(runner, "speculative_config", None)
    return (
        getattr(spec_config, "method", None) == "mtp"
        and getattr(runner, "num_spec_tokens", 0) > 0
    )


def _deepseek_v4_mtp_uniform_decode_warmup_requests(
    runner: "GPUModelRunner",
    max_tokens: int,
    max_reqs: int,
) -> tuple[int, ...]:
    if not _is_deepseek_v4_mtp_spec_decode(runner):
        return ()

    query_len = getattr(
        runner,
        "uniform_decode_query_len",
        1 + getattr(runner, "num_spec_tokens", 0),
    )
    if query_len <= 0:
        return ()

    max_warmup_reqs = min(
        max_reqs,
        max_tokens // query_len,
        _DEEPSEEK_V4_MTP_UNIFORM_DECODE_MAX_WARMUP_REQUESTS,
    )
    candidates = sorted(
        set(_DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS)
        | {max_warmup_reqs}
    )
    return tuple(reqs for reqs in candidates if reqs <= max_warmup_reqs)


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    max_tokens = getattr(runner, "max_num_tokens", 1)
    block_table = runner.input_batch.block_table

    # Snapshot the runner buffers we mutate so warmup never leaks state into
    # the first real request.
    saved_query_start_loc_np: np.ndarray | None = None
    saved_query_start_loc_gpu: torch.Tensor | None = None
    if hasattr(runner, "query_start_loc"):
        saved_query_start_loc_np = runner.query_start_loc.np[:2].copy()
        saved_query_start_loc_gpu = runner.query_start_loc.gpu[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if hasattr(runner, "query_start_loc"):
                runner.query_start_loc.np[0] = 0
                runner.query_start_loc.np[1] = num_tokens
                runner.query_start_loc.copy_to_gpu(2)
                query_start_loc = runner.query_start_loc.gpu[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            if hasattr(runner, "positions"):
                saved_positions: torch.Tensor | None = (
                    runner.positions[:num_tokens].clone()
                )
                runner.positions[:num_tokens].copy_(positions_source)
                positions = runner.positions[:num_tokens]
            else:
                saved_positions = None
                positions = positions_source

            try:
                block_table.commit_block_table(1)
                block_table.compute_slot_mapping(1, query_start_loc, positions)
            finally:
                if saved_positions is not None:
                    runner.positions[:num_tokens].copy_(saved_positions)
    finally:
        if saved_query_start_loc_np is not None:
            runner.query_start_loc.np[:2] = saved_query_start_loc_np
            assert saved_query_start_loc_gpu is not None
            runner.query_start_loc.gpu[:2].copy_(saved_query_start_loc_gpu)


def _deepseek_v4_structured_output_bitmask_warmup(
    runner: "GPUModelRunner",
) -> None:
    vocab_size = runner.model_config.get_vocab_size()
    if vocab_size <= 0:
        return

    dtypes = [torch.float32]
    model_dtype = getattr(runner.model_config, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype not in dtypes:
        dtypes.append(model_dtype)

    bitmask_width = (vocab_size + 31) // 32
    req_id = "_deepseek_v4_warmup_"
    grammar_bitmask = np.full((1, bitmask_width), fill_value=-1, dtype=np.int32)
    grammar_output = GrammarOutput(
        structured_output_request_ids=[req_id], grammar_bitmask=grammar_bitmask
    )

    for dtype in dtypes:
        for req_ids in ([req_id], [req_id, "_deepseek_v4_warmup_unmasked_"]):
            logits = torch.zeros(
                (len(req_ids), vocab_size), dtype=dtype, device=runner.device
            )
            input_batch = SimpleNamespace(req_ids=req_ids)
            apply_grammar_bitmask(
                SchedulerOutput.make_empty(), grammar_output, input_batch, logits
            )


def _deepseek_v4_indexer_shape(runner: "GPUModelRunner") -> tuple[int, int, int]:
    hf_config = runner.model_config.hf_config
    num_heads = int(getattr(hf_config, "index_n_heads", 64))
    head_dim = int(getattr(hf_config, "index_head_dim", 128))
    topk = int(getattr(hf_config, "index_topk", 512))
    return num_heads, head_dim, topk


def _deepseek_v4_compress_ratios(runner: "GPUModelRunner") -> tuple[int, ...]:
    hf_config = runner.model_config.hf_config
    ratios = getattr(hf_config, "compress_ratios", None) or (128,)
    return tuple(sorted({int(r) for r in ratios if int(r) > 0}))


@torch.inference_mode()
def _deepseek_v4_row_tiled_logits_warmup(runner: "GPUModelRunner") -> None:
    if not (
        envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP
        and envs.VLLM_SM12X_MQA_TOPK_TRITON
        and envs.VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED
        and current_platform.is_cuda_alike()
    ):
        return

    num_heads, head_dim, topk = _deepseek_v4_indexer_shape(runner)
    if topk not in (512, 2048):
        return

    row_tile = min(
        max(1, envs.VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE),
        max(1, worker_max_tokens := runner.max_num_tokens),
    )
    min_kv_tokens = max(1, envs.VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS)
    kv_tokens = min(max(min_kv_tokens, topk * 16), worker_max_tokens)
    if kv_tokens < min_kv_tokens:
        return

    try:
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import (
            fp8_fp4_mqa_topk_indices,
        )

        device = runner.device
        q = torch.zeros(
            (row_tile, num_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        k = torch.zeros((kv_tokens, head_dim), dtype=torch.float8_e4m3fn, device=device)
        scale = torch.ones((kv_tokens,), dtype=torch.float32, device=device)
        weights = torch.full(
            (row_tile, num_heads),
            1.0 / max(1, num_heads),
            dtype=torch.float32,
            device=device,
        )
        cu_seqlen_ks = torch.zeros(row_tile, dtype=torch.int32, device=device)
        cu_seqlen_ke = torch.full(
            (row_tile,), kv_tokens, dtype=torch.int32, device=device
        )
        out = torch.empty((row_tile, topk), dtype=torch.int32, device=device)

        logger.info(
            "Warming up DeepSeek V4 row-tiled MQA logits top-k "
            "(rows=%d, kv_tokens=%d, heads=%d, head_dim=%d, topk=%d).",
            row_tile,
            kv_tokens,
            num_heads,
            head_dim,
            topk,
        )
        used = fp8_fp4_mqa_topk_indices(
            (q, None),
            (k, scale),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out,
        )
        if not used:
            logger.warning_once(
                "DeepSeek V4 row-tiled MQA logits top-k warmup did not use "
                "the Triton path."
            )
    except Exception:
        logger.warning_once(
            "DeepSeek V4 row-tiled MQA logits top-k warmup failed.",
            exc_info=True,
        )


@torch.inference_mode()
def _deepseek_v4_sparse_mla_metadata_warmup(runner: "GPUModelRunner") -> None:
    if not (
        envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP
        and current_platform.is_cuda_alike()
    ):
        return

    try:
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            build_c128a_topk_metadata,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            build_prefill_chunk_metadata,
        )
        from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
            combine_topk_swa_indices,
        )

        device = runner.device
        block_size = int(getattr(runner.cache_config, "block_size", None) or 256)
        _, _, topk = _deepseek_v4_indexer_shape(runner)
        max_compressed_tokens = max(topk, 8192)
        window_size = int(
            getattr(runner.model_config.hf_config, "sliding_window", None) or 128
        )
        num_tokens = min(_DEEPSEEK_V4_METADATA_WARMUP_TOKENS, runner.max_num_tokens)
        if num_tokens <= 0:
            return

        logger.info(
            "Warming up DeepSeek V4 sparse MLA metadata kernels for "
            "compress_ratios=%s.",
            list(_deepseek_v4_compress_ratios(runner)),
        )
        for compress_ratio in _deepseek_v4_compress_ratios(runner):
            warm_pos = compress_ratio * max_compressed_tokens - 1
            positions = torch.full(
                (num_tokens,), warm_pos, dtype=torch.int64, device=device
            )
            token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
            block_columns = warm_pos // block_size + 2
            block_table = torch.zeros((1, block_columns), dtype=torch.int32, device=device)
            slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=device)

            global_decode_buffer = torch.empty(
                (num_tokens, max_compressed_tokens), dtype=torch.int32, device=device
            )
            decode_lens_buffer = torch.empty(num_tokens, dtype=torch.int32, device=device)
            prefill_buffer = torch.empty(
                (num_tokens, max_compressed_tokens), dtype=torch.int32, device=device
            )
            build_c128a_topk_metadata(
                positions,
                compress_ratio,
                0,
                token_to_req,
                block_table,
                block_size,
                slot_mapping,
                global_decode_buffer,
                decode_lens_buffer,
                prefill_buffer,
                max_compressed_tokens=max_compressed_tokens,
            )
            build_c128a_topk_metadata(
                positions,
                compress_ratio,
                min(2, num_tokens),
                token_to_req,
                block_table,
                block_size,
                slot_mapping,
                global_decode_buffer,
                decode_lens_buffer,
                prefill_buffer,
                max_compressed_tokens=max_compressed_tokens,
            )

            query_len = min(8192, runner.max_num_tokens)
            query_start_loc_cpu = torch.tensor([0, query_len], dtype=torch.int32)
            query_start_loc = query_start_loc_cpu.to(device)
            uncompressed_seq_lens = torch.tensor(
                [query_len * 2], dtype=torch.int32, device=device
            )
            compressed_seq_lens_cpu = torch.tensor(
                [max(1, (query_len * 2) // compress_ratio)], dtype=torch.int32
            )
            compressed_seq_lens = compressed_seq_lens_cpu.to(device)
            build_prefill_chunk_metadata(
                0,
                1,
                query_start_loc,
                query_start_loc_cpu,
                uncompressed_seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                compress_ratio,
                query_slice=slice(0, query_len),
                skip_kv_gather=True,
            )

            topk_indices = torch.full(
                (num_tokens, topk), -1, dtype=torch.int32, device=device
            )
            topk_indices[:, : min(topk, 16)] = torch.arange(
                min(topk, 16), dtype=torch.int32, device=device
            )
            query_start_loc_small = torch.tensor(
                [0, num_tokens], dtype=torch.int32, device=device
            )
            seq_lens = torch.full(
                (1,), query_len * 2, dtype=torch.int32, device=device
            )
            gather_lens = torch.full((1,), query_len, dtype=torch.int32, device=device)
            combine_topk_swa_indices(
                topk_indices,
                query_start_loc_small,
                seq_lens,
                gather_lens,
                window_size,
                compress_ratio,
                topk,
                num_tokens,
                topk,
            )
    except Exception:
        logger.warning_once(
            "DeepSeek V4 sparse MLA metadata warmup failed.",
            exc_info=True,
        )


@torch.inference_mode()
def _deepseek_v4_sparse_mla_swa_decode_warmup(runner: "GPUModelRunner") -> None:
    if not (
        envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP
        and current_platform.is_cuda_alike()
    ):
        return

    try:
        from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
            fp8ds_paged_sparse_mla_attention_with_sink_multihead,
            sparse_mla_decode_head_block_size,
        )

        hf_config = runner.model_config.hf_config
        device = runner.device
        block_size = int(getattr(runner.cache_config, "block_size", None) or 256)
        num_heads, _, _ = _deepseek_v4_indexer_shape(runner)
        head_dim = 512
        token_fp8_dim = 448
        token_bf16_dim = 64
        token_scale_dim = 8
        token_data_size = token_fp8_dim + token_bf16_dim * 2
        window_size = int(getattr(hf_config, "sliding_window", None) or 128)
        num_decode_tokens = 1
        num_candidates = min(max(1, window_size), block_size)

        q = torch.zeros(
            (num_decode_tokens, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        k_cache = torch.zeros(
            (1, block_size, token_data_size + token_scale_dim),
            dtype=torch.uint8,
            device=device,
        )
        seq_lens = torch.full(
            (num_decode_tokens,), num_candidates, dtype=torch.int32, device=device
        )
        gather_lens = torch.full_like(seq_lens, num_candidates)
        block_table = torch.zeros((num_decode_tokens, 1), dtype=torch.int32, device=device)
        attn_sink = torch.full((num_heads,), -float("inf"), dtype=torch.float32, device=device)
        output = torch.empty(
            (num_decode_tokens, num_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )

        logger.info(
            "Warming up DeepSeek V4 sparse MLA SWA decode kernel "
            "(tokens=%d, candidates=%d, heads=%d).",
            num_decode_tokens,
            num_candidates,
            num_heads,
        )
        fp8ds_paged_sparse_mla_attention_with_sink_multihead(
            q=q,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            candidate_offset=0,
            num_candidates=num_candidates,
            scale=1.0,
            attn_sink=attn_sink,
            output=output,
            head_block_size=sparse_mla_decode_head_block_size(num_decode_tokens),
            num_heads=num_heads,
        )
    except Exception:
        logger.warning_once(
            "DeepSeek V4 sparse MLA SWA decode warmup failed.",
            exc_info=True,
        )


@torch.inference_mode()
def _deepseek_v4_flashinfer_b12x_route_pack_warmup(
    runner: "GPUModelRunner",
) -> None:
    if not (
        envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP
        and envs.VLLM_USE_FLASHINFER_MOE_B12X_W4A16
        and current_platform.is_cuda_alike()
    ):
        return

    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
            select_route_block_size_m,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_route_pack import (
            pack_topk_routes_by_expert,
        )
        from vllm.utils.flashinfer import has_flashinfer_b12x_fused_moe

        if not has_flashinfer_b12x_fused_moe():
            return

        hf_config = runner.model_config.hf_config
        topk = int(getattr(hf_config, "num_experts_per_tok", 6))
        num_experts = int(getattr(hf_config, "n_routed_experts", 256))
        token_counts = tuple(
            sorted(
                {
                    t
                    for t in (
                        *_DEEPSEEK_V4_ROUTE_PACK_WARMUP_TOKENS,
                        runner.max_num_tokens,
                    )
                    if t > 0
                }
            )
        )

        device = runner.device
        expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)
        logger.info(
            "Warming up FlashInfer B12x W4A16 route-pack kernels for "
            "token_counts=%s.",
            list(token_counts),
        )
        for num_tokens in token_counts:
            topk_ids = (
                torch.arange(num_tokens * topk, dtype=torch.int32, device=device)
                .remainder(num_experts)
                .reshape(num_tokens, topk)
            )
            block_size = select_route_block_size_m(num_tokens, topk, num_experts)
            pack_topk_routes_by_expert(
                topk_ids,
                block_size,
                num_experts,
                expert_map=expert_map,
            )
            pack_topk_routes_by_expert(
                topk_ids,
                block_size,
                num_experts,
                expert_map=None,
            )
    except Exception:
        logger.warning_once(
            "FlashInfer B12x W4A16 route-pack warmup failed.",
            exc_info=True,
        )


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)

    if getattr(runner, "is_last_pp_rank", True):
        try:
            _deepseek_v4_structured_output_bitmask_warmup(runner)
        except ImportError:
            logger.debug(
                "Skipping DeepSeek V4 structured output bitmask warmup because "
                "xgrammar is unavailable."
            )

    torch.accelerator.synchronize()


def _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
    *,
    device: torch.device,
    num_reqs: int,
    num_spec_tokens: int,
    vocab_size: int,
    block_size: int,
    max_model_len: int,
) -> None:
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import rejection_sample
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_update_slot_mapping_and_metadata,
        next_power_of_2,
    )

    num_sampled_tokens = num_spec_tokens + 1
    sampled_token_ids = torch.arange(
        num_reqs * num_sampled_tokens, dtype=torch.int32, device=device
    ).reshape(num_reqs, num_sampled_tokens)
    sampled_token_ids.remainder_(vocab_size)
    discard_request_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
    backup_next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
    valid_sampled_tokens_count = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
        num_sampled_tokens,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled_tokens),
    )

    cu_num_draft_tokens = torch.arange(
        num_spec_tokens,
        num_reqs * num_spec_tokens + 1,
        num_spec_tokens,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.arange(
        0,
        (num_reqs + 1) * num_sampled_tokens,
        num_sampled_tokens,
        dtype=torch.int32,
        device=device,
    )
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens,
        num_reqs,
    )

    positions = torch.arange(num_reqs, dtype=torch.int64, device=device)
    block_table_tensor = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    out_clamped_positions = torch.empty_like(positions)
    out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=device)
    eagle_step_update_slot_mapping_and_metadata(
        positions,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size=num_reqs,
    )

    total_draft_tokens = num_reqs * num_spec_tokens
    draft_token_ids = torch.arange(total_draft_tokens, dtype=torch.int32, device=device)
    draft_token_ids.remainder_(vocab_size)
    draft_probs = torch.rand(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_logits = torch.randn(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    bonus_token_ids = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    sampling_metadata = SamplingMetadata(
        temperature=torch.full((num_reqs,), 0.7, dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=device),
        presence_penalties=torch.empty(0, device=device),
        repetition_penalties=torch.empty(0, device=device),
        output_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=[[] for _ in range(num_reqs)],
    )
    rejection_sample(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[num_spec_tokens] * num_reqs,
        max_spec_len=num_spec_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
    )


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    prefill_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS, max_tokens
    )
    uniform_decode_reqs = _deepseek_v4_mtp_uniform_decode_warmup_requests(
        runner,
        max_tokens=max_tokens,
        max_reqs=worker.scheduler_config.max_num_seqs,
    )
    if mixed_tokens <= 0 and prefill_tokens <= 0 and not uniform_decode_reqs:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed tokens=%s, prefill tokens=%s, and MTP uniform decode "
        "requests=%s.",
        mixed_tokens,
        prefill_tokens,
        list(uniform_decode_reqs),
    )
    if mixed_tokens > 0:
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    if prefill_tokens > 0:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
        # Simulate the second-and-later chunk of a chunked prefill so
        # `_build_prefill_chunk_metadata_kernel` and the alt-shape
        # `_w8a8_triton_block_scaled_mm` configs that fire when the
        # indexer sees prior context get JIT-compiled here, not on the
        # first user request that exceeds `max_num_batched_tokens`.
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
            profile_seq_lens=prefill_tokens * 2,
        )
        # NOTE: The multi-request prefill warmup that previously sat here
        # (max_num_seqs prefills sharing the batched-token budget) hit a
        # CUDA illegal memory access inside the CUTeDSL
        # ``DequantGatherKCacheKernel`` on SM12x. The dummy_run shape it
        # generated violates an implicit ``offset + gather_len <= M``
        # invariant of the kv-gather output buffer (M is sized for the
        # single-prefill warmup case). Removing the warmup gives back the
        # one-time JIT cost on the first real multi-prefill request, but
        # unblocks serve startup at production ``--max-num-seqs`` values
        # (e.g. 128). Re-enable once the gather-buffer sizing for
        # multi-request prefill warmup is reconciled with the kernel's
        # bounds.
    query_len = getattr(runner, "uniform_decode_query_len", 0)
    for num_reqs in uniform_decode_reqs:
        runner._dummy_run(
            num_tokens=num_reqs * query_len,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
        )

    if uniform_decode_reqs and current_platform.is_cuda_alike():
        vocab_size = runner.model_config.get_vocab_size()
        block_size = getattr(runner.cache_config, "block_size", None) or 16
        logger.info(
            "Warming up DeepSeek V4 MTP spec-decode kernels for request "
            "counts=%s and %d draft tokens.",
            list(uniform_decode_reqs),
            runner.num_spec_tokens,
        )
        for num_reqs in uniform_decode_reqs:
            _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
                device=runner.device,
                num_reqs=num_reqs,
                num_spec_tokens=runner.num_spec_tokens,
                vocab_size=vocab_size,
                block_size=block_size,
                max_model_len=runner.max_model_len,
            )
        torch.accelerator.synchronize()


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_row_tiled_logits_warmup(worker.model_runner)
    _deepseek_v4_sparse_mla_metadata_warmup(worker.model_runner)
    _deepseek_v4_sparse_mla_swa_decode_warmup(worker.model_runner)
    _deepseek_v4_flashinfer_b12x_route_pack_warmup(worker.model_runner)
    _deepseek_v4_request_prep_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    import vllm.utils.flashinfer as fi_utils

    with torch.inference_mode(), fi_utils.autotune():
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False
