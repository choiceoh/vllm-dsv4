# DeepSeek V4 Flash GB10 Row-Tiled Logits Baseline Setup

This baseline captures the working two-node GB10 setup validated on
2026-05-19. It is meant to be a stable starting image for future agents so they
can build one baseline image and then layer experiments on top of that image.

## What This Baseline Contains

- vLLM branch: `production-baseline-20260515`
- vLLM source savepoint: the commit or worktree state that contains this file,
  the row-tiled logits change in `sm12x_deep_gemm_fallbacks.py`, and the
  baseline Dockerfile update, descending from
  `5b5b63ded docs: checkpoint flashinfer static mla baseline`
- FlashInfer checkout: `/home/aidendle94/Documents/workspace/flashinfer-latest`
- FlashInfer branch: `pr-3336-w4a16-rewrite`
- FlashInfer commit: `a52ad3a649ab3716efe738be24e837566117d2b3`
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Runtime shape: TP=2, PP=1, EP enabled, MTP=2, fp8 KV
- Context shape: `max_model_len=262144`, `max_num_batched_tokens=8192`, `max_num_seqs=1`
- MoE backend: `FLASHINFER_B12X_MXFP4_BF16`
- Static MLA/MQA path: `VLLM_SM12X_MQA_TOPK_TRITON=1`
- MQA top-k baseline: row-tiled materialized logits enabled with
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED=1` and
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE=512`
- Combined prefill opts:
  `VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS=512`,
  `VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES=2`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H=8`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D=32`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS=4`,
  `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=2048`, and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE=1024`
- Blocked sparse MLA prefill accumulator enabled in the baseline image and
  recipe:
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM=1`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE=1`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C=16`, and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS=8`

The source-level blocked accumulator knob remains default-off. This baseline
opts in through the Dockerfile and launch recipe after service-level validation.

The current row-tiled logits warm measurements are:

| context | max output | measured prefill tok/s | measured decode tok/s | prefill seconds | prefix hits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32k | 1 | 1233.11 | n/a | 25.95 | 0 |
| 128k | 1 | 961.68 | n/a | 133.10 | 0 |
| 128k | 64 | 938.33 | 26.48 | 136.42 | 0 |

The previous H8/D32/W4 streaming baseline warm measured prefill ruler was:

| context | measured prefill tok/s | prefill seconds |
| ---: | ---: | ---: |
| 1k | 1146.11 | 0.87 |
| 4k | 1318.31 | 3.03 |
| 8k | 1274.45 | 6.28 |
| 16k | 1270.87 | 12.59 |
| 32k | 1232.91 | 25.95 |
| 64k | 1000.59 | 63.96 |
| 96k | 844.10 | 113.73 |
| 128k | 750.71 | 170.51 |
| 160k | 690.99 | 231.55 |
| 200k | 665.31 | 300.61 |

All measured ruler legs used `max_tokens=1`, returned status 200, had prefix
cache hits delta `0`, and reported computed prefill tokens equal to the prompt
size.

Compared to the prior hybrid MQA logits 200k profile baseline, the H8/D32/W4
topk512 candidate improved the warm profiled 200k request from `609.22 tok/s`
to `632.84 tok/s` and reduced rank0 `_fp8_mqa_topk_stream_kernel` time by
`17.04%`. The no-profiler ruler measured `665.31 tok/s` at 200k.

The same live baseline server also passed the long-context haystack retrieval
probe at both 128k and 200k:

| target | prompt lines | prompt tokens | completion tokens | elapsed seconds | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128k | 4225 | 131032 | 69 | 198.78 | matched all sentinel terms |
| 200k | 6455 | 200162 | 52 | 319.07 | matched all sentinel terms |

The 128k haystack run was the first post-launch long request and showed
first-use Triton JIT warnings for metadata, logits, top-k, attention, and MoE
kernels. The 200k haystack was run immediately afterward on the same server.

This promotes row512/q2048 plus blocked sparse MLA accumulation plus row-tiled
materialized MQA logits top-k as the baseline. The optimization does not reduce
`topk`; it computes exact topk512 over full logits in row tiles, with the
streaming H8/D32/W4 path kept as the fallback for non-eligible shapes.

The full benchmark artifact is:

```text
/home/aidendle94/Documents/workspace/experiment_runs/blocked_accum_20260518T092400Z/report.md
/home/aidendle94/Documents/workspace/experiment_runs/topk_h8d32w4_profile_200k_20260519T014917Z/report.md
/home/aidendle94/Documents/workspace/experiment_runs/topk_h8d32w4_prefill_ruler_20260519T024613Z/report.md
/home/aidendle94/Documents/workspace/experiment_runs/topk_h8d32w4_haystack_20260519T034225Z/128k/long_context_probe.md
/home/aidendle94/Documents/workspace/experiment_runs/topk_h8d32w4_haystack_20260519T034225Z/200k/long_context_probe.md
/home/aidendle94/Documents/workspace/experiment_runs/mqa_logits_rowtile_20260519T070940Z/runs/rowtile_32000/warm_results.json
/home/aidendle94/Documents/workspace/experiment_runs/mqa_logits_rowtile_20260519T070940Z/runs/rowtile_128000/warm_results.json
/home/aidendle94/Documents/workspace/experiment_runs/rowtiled_logits_baseline_20260519T081248Z/runs/rowtiled_baseline_128000_decode64/warm_results.json
```

## Build The Baseline Image

Build from the workspace root, not from the vLLM root. The Dockerfile needs the
sibling `flashinfer-latest` checkout in the same build context.

```bash
set -euo pipefail

export WORKSPACE=/home/aidendle94/Documents/workspace
export VLLM_ROOT=${WORKSPACE}/vllm
export BASE_IMAGE=sparkrun-vllm-ds4-gb10:ae353d502-static-mla-dirty-20260518T000139Z-cuda13.2-nccl2.30-vllm-openai-base
export BASELINE_REF=$(git -C "${VLLM_ROOT}" rev-parse --short HEAD)
export BASELINE_IMAGE=sparkrun-vllm-ds4-gb10:${BASELINE_REF}-rowtiled-logits-baseline-cuda13.2-nccl2.30-vllm-openai-base

git -C "${VLLM_ROOT}" merge-base --is-ancestor 5b5b63ded HEAD
test "$(git -C "${WORKSPACE}/flashinfer-latest" rev-parse HEAD)" = "a52ad3a649ab3716efe738be24e837566117d2b3"

DOCKER_BUILDKIT=1 docker build \
  --file "${VLLM_ROOT}/Dockerfile.baseline" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${BASELINE_IMAGE}" \
  "${WORKSPACE}"
```

The build-time assertion checks that vLLM can see the FlashInfer B12x W4A16
entrypoints. The Dockerfile also bakes the row-tiled logits top-k env defaults,
row512/q2048 runtime env defaults, H8/D32/W4 stream fallback env defaults, and
blocked sparse MLA accumulator env defaults listed in the recipe below, so
future overlay images inherit this baseline even when the launch recipe does
not restate every knob.

```python
from vllm.utils.flashinfer import has_flashinfer_b12x_fused_moe
assert has_flashinfer_b12x_fused_moe()
```

Use `${BASELINE_IMAGE}` as the parent image for future experiment Dockerfiles.

## Launch Recipe

Write this recipe to a temporary file or to the harness repo. Replace the image
only when intentionally testing a descendant image.

```yaml
recipe_version: "1"
name: DeepSeek V4 Flash GB10 row-tiled logits prefill baseline
description: DeepSeek V4 Flash GB10 baseline with FlashInfer B12x W4A16, static SM12x MQA top-k, row-tiled materialized logits top-k, blocked sparse MLA prefill accumulation, combined prefill opts, MTP=2, and 262k context.
runtime: vllm-distributed
model: deepseek-ai/DeepSeek-V4-Flash
container: sparkrun-vllm-ds4-gb10:<commit>-rowtiled-logits-baseline-cuda13.2-nccl2.30-vllm-openai-base
cluster_only: true
min_nodes: 2
max_nodes: 2

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  pipeline_parallel: 1
  gpu_memory_utilization: 0.85
  max_model_len: 262144
  max_num_batched_tokens: 8192
  max_num_seqs: 1
  block_size: 256
  kv_cache_dtype: fp8
  served_model_name: deepseek-v4-flash

env:
  TORCH_CUDA_ARCH_LIST: 12.1a
  VLLM_ENFORCE_STRICT_TOOL_CALLING: "1"
  VLLM_USE_FLASHINFER_MOE_B12X_W4A16: "1"
  VLLM_SM12X_MQA_TOPK_TRITON: "1"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED: "1"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE: "512"
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS: "64"
  VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS: "512"
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS: "8192"
  VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES: "2"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H: "8"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D: "32"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS: "4"
  VLLM_TRITON_MLA_SPARSE: "1"
  VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: "2048"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE: "1024"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C: "16"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS: "8"
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
  VLLM_NCCL_SO_PATH: /usr/lib/aarch64-linux-gnu/libnccl.so.2
  NCCL_IB_DISABLE: "0"
  NCCL_DEBUG: WARN

command: |
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name {served_model_name} \
    --host {host} \
    --port {port} \
    --trust-remote-code \
    --tensor-parallel-size {tensor_parallel} \
    --pipeline-parallel-size {pipeline_parallel} \
    --enable-expert-parallel \
    --kv-cache-dtype {kv_cache_dtype} \
    --block-size {block_size} \
    --max-model-len {max_model_len} \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --distributed-executor-backend mp \
    --no-enable-flashinfer-autotune \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
```

Launch it:

```bash
cd /home/aidendle94/Documents/workspace/vllm-ds4-sm120-harness
sparkrun run /tmp/deepseek-v4-flash-gb10-rowtiled-logits-baseline.yaml --no-follow
```

If reusing an older checked harness recipe, make sure it includes the
combined-prefill and blocked-accumulator env vars above, uses `MAX_ROWS=512`,
`QUERY_CHUNK_SIZE=2048`, `PREFILL_BLOCK_C=16`, `PREFILL_BLOCK_HEADS=8`, and
the row-tiled logits and H8/D32/W4 fallback topk512 env vars, and removes any
`--profiler-config` unless you
are intentionally taking a profile.
The old static-MLA profile recipe is not the clean throughput baseline by
itself.

```bash
rg "LOGITS_ROW|MAX_ROWS|STREAM_K_TILES|TOPK512|QUERY_CHUNK|PREFILL_TOPK_CHUNK|BLOCKED_ACCUM|BLOCK_C|BLOCK_HEADS|profiler-config" sparkrun/*.yaml
```

## Validation

Health and model checks:

```bash
curl -fsS --max-time 10 http://192.168.50.29:8000/health
curl -fsS --max-time 10 http://192.168.50.29:8000/v1/models
```

Required log evidence:

```bash
sparkrun logs <cluster-id> --tail 300 | rg \
  "FLASHINFER_B12X|materialized MQA logits top-k row-tiled|Started server process|GET /health"
```

Expected evidence:

```text
Using 'FLASHINFER_B12X_MXFP4_BF16' Mxfp4 MoE backend.
Using SM12x Triton materialized MQA logits top-k row-tiled path (... row_tile=512 ...)
GET /health HTTP/1.1" 200 OK
```

Required recipe evidence:

```bash
sparkrun export running-recipe <cluster-id> | rg \
  "BLOCKED_ACCUM|PREFILL_BLOCK_C|PREFILL_BLOCK_HEADS|QUERY_CHUNK_SIZE|LOGITS_ROW|MAX_ROWS|TOPK512"
```

Expected recipe evidence:

```text
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM: '1'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE: '1'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C: '16'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS: '8'
VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: '2048'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED: '1'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE: '512'
VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS: '512'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H: '8'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D: '32'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS: '4'
```

The first long request may still show Triton JIT warnings for row-tiled logits,
metadata, sparse MLA, MoE, or decode kernels if that exact shape was not covered
by warmup. Treat the second request at the same context length as the warm
measurement.

For 32k, 64k, and 128k benchmarks, use an uncached single request with exact
prompt token IDs and 128 output tokens, then compute throughput from Prometheus
metric deltas:

```text
server_prefill_tok_per_s = request_prefill_kv_computed_tokens_delta / request_prefill_time_seconds_delta
server_decode_tok_per_s = generation_tokens_delta / request_decode_time_seconds_delta
```

Do not use warmed benchmark-average input-rate math as the prefill metric when
prefix cache could be involved. Confirm `vllm:prefix_cache_hits_total` delta is
zero for the measured request.
