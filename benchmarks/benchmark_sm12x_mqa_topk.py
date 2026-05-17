# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark SM12x DeepSeek V4 MQA prefill top-k paths.

This benchmark uses synthetic tensors with the same shapes consumed by
``SparseAttnIndexer``. It does not launch a model.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import (  # noqa: E402
    _fp8_mqa_logits_topk_torch,
)
from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (  # noqa: E402
    fp8_mqa_topk_indices_triton,
)


def _make_tensors(
    rows: int,
    kv_tokens: int,
    heads: int,
    head_dim: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(
        rows,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn).contiguous()
    kv = torch.randn(kv_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_scale = kv.abs().float().amax(dim=-1).clamp(1e-4) / 448.0
    kv_fp8 = (kv * kv_scale.reciprocal()[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.float32).contiguous()
    cu_seqlen_ks = torch.zeros(rows, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.full(
        (rows,), kv_tokens, device="cuda", dtype=torch.int32
    )
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    return (
        q_fp8,
        kv_fp8.contiguous(),
        kv_scale.contiguous(),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
    )


def _time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _validate(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    # Sparse MLA consumes the selected candidate set; order differences from
    # different top-k tie handling are not semantically relevant here.
    actual_sorted = torch.sort(actual, dim=1).values
    expected_sorted = torch.sort(expected, dim=1).values
    if not torch.equal(actual_sorted, expected_sorted):
        mismatches = int((actual_sorted != expected_sorted).sum().item())
        raise AssertionError(f"top-k set mismatch: {mismatches} entries differ")


def run_case(args: argparse.Namespace) -> dict[str, object]:
    (
        q,
        kv,
        kv_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
    ) = _make_tensors(
        args.rows,
        args.kv_tokens,
        args.heads,
        args.head_dim,
        args.topk,
        args.seed,
    )
    expected = torch.empty_like(out)

    def torch_path() -> None:
        _fp8_mqa_logits_topk_torch(
            (q, None),
            (kv, kv_scale),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            args.topk,
            out=expected,
        )

    def triton_path() -> None:
        ok = fp8_mqa_topk_indices_triton(
            q,
            (kv, kv_scale),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out,
        )
        if not ok:
            raise RuntimeError("Triton MQA top-k path rejected this shape")

    if args.validate:
        torch_path()
        triton_path()
        torch.cuda.synchronize()
        _validate(out, expected)

    triton_ms = _time_cuda(triton_path, args.warmup, args.iters)
    torch_ms = _time_cuda(torch_path, args.warmup, args.iters)
    result = {
        "rows": args.rows,
        "kv_tokens": args.kv_tokens,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "topk": args.topk,
        "triton_avg_ms": statistics.mean(triton_ms),
        "triton_min_ms": min(triton_ms),
        "torch_avg_ms": statistics.mean(torch_ms),
        "torch_min_ms": min(torch_ms),
        "speedup_vs_torch_avg": statistics.mean(torch_ms) / statistics.mean(triton_ms),
        "speedup_vs_torch_min": min(torch_ms) / min(triton_ms),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--kv-tokens", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, choices=(512, 2048), default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction,
                        default=False)
    return parser.parse_args()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1
    result = run_case(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
