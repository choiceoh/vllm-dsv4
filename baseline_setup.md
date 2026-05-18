# DeepSeek V4 Flash GB10 Combined Prefill Baseline Setup

This baseline captures the working two-node GB10 setup validated on
2026-05-18. It is meant to be a stable starting image for future agents so they
can build one baseline image and then layer experiments on top of that image.

## What This Baseline Contains

- vLLM branch: `production-baseline-20260515`
- vLLM source savepoint: the commit that contains this file, descending from
  `5b5b63ded docs: checkpoint flashinfer static mla baseline`
- FlashInfer checkout: `/home/aidendle94/Documents/workspace/flashinfer-latest`
- FlashInfer branch: `pr-3336-w4a16-rewrite`
- FlashInfer commit: `a52ad3a649ab3716efe738be24e837566117d2b3`
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Runtime shape: TP=2, PP=1, EP enabled, MTP=2, fp8 KV
- Context shape: `max_model_len=131072`, `max_num_batched_tokens=8192`, `max_num_seqs=1`
- MoE backend: `FLASHINFER_B12X_MXFP4_BF16`
- Static MLA/MQA path: `VLLM_SM12X_MQA_TOPK_TRITON=1`
- Combined prefill opts:
  `VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS=512`,
  `VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES=2`,
  `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=2048`, and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE=1024`

The validated warm measured results for this setup were:

- 32k request: `454.57 tok/s` server prefill, `44.74 tok/s` server decode
- 128k request: `559.72 tok/s` server prefill, `37.13 tok/s` server decode
- Prefix cache hits: `0` for every warmup and measured leg

Compared to the previous committed combined-prefill recipe
(`row_tile=256`, `query_chunk=1024`):

- 32k measured prefill: `-2.62%`; decode: `+11.37%`
- 128k measured prefill: `+8.52%`; decode: `+9.98%`

This is an intentional long-context baseline tradeoff: keep the row512/q2048
recipe because 128k prefill and decode improve materially, while the measured
32k prefill dip is small enough for this baseline.

The full benchmark artifact is:

```text
/home/aidendle94/Documents/workspace/experiment_runs/aggressive_row512_q2048_32k_128k_20260518T074836Z/report.md
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
export BASELINE_IMAGE=sparkrun-vllm-ds4-gb10:${BASELINE_REF}-combined-prefill-baseline-cuda13.2-nccl2.30-vllm-openai-base

git -C "${VLLM_ROOT}" merge-base --is-ancestor 5b5b63ded HEAD
test "$(git -C "${WORKSPACE}/flashinfer-latest" rev-parse HEAD)" = "a52ad3a649ab3716efe738be24e837566117d2b3"

DOCKER_BUILDKIT=1 docker build \
  --file "${VLLM_ROOT}/Dockerfile.baseline" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${BASELINE_IMAGE}" \
  "${WORKSPACE}"
```

The build-time assertion checks that vLLM can see the FlashInfer B12x W4A16
entrypoints:

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
name: DeepSeek V4 Flash GB10 combined prefill baseline
description: DeepSeek V4 Flash GB10 baseline with FlashInfer B12x W4A16, static SM12x MQA top-k, combined prefill opts, MTP=2, and 128k context.
runtime: vllm-distributed
model: deepseek-ai/DeepSeek-V4-Flash
container: sparkrun-vllm-ds4-gb10:<commit>-combined-prefill-baseline-cuda13.2-nccl2.30-vllm-openai-base
cluster_only: true
min_nodes: 2
max_nodes: 2

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  pipeline_parallel: 1
  gpu_memory_utilization: 0.85
  max_model_len: 131072
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
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS: "64"
  VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS: "512"
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS: "8192"
  VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES: "2"
  VLLM_TRITON_MLA_SPARSE: "1"
  VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: "2048"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE: "1024"
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
sparkrun run /tmp/deepseek-v4-flash-gb10-combined-prefill-baseline.yaml --no-follow
```

If reusing an older checked harness recipe, make sure it includes the
combined-prefill env vars above, uses `MAX_ROWS=512` and
`QUERY_CHUNK_SIZE=2048`, and removes any `--profiler-config` unless you are
intentionally taking a profile. The old static-MLA profile recipe is not the
clean throughput baseline by itself.

```bash
rg "MAX_ROWS|STREAM_K_TILES|QUERY_CHUNK|PREFILL_TOPK_CHUNK|profiler-config" sparkrun/*.yaml
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
  "FLASHINFER_B12X|Using SM12x Triton prefill MQA|Started server process|GET /health"
```

Expected evidence:

```text
Using 'FLASHINFER_B12X_MXFP4_BF16' Mxfp4 MoE backend.
Using SM12x Triton prefill MQA top-k path (... row_tile=512 ...)
GET /health HTTP/1.1" 200 OK
```

The first long request may still show a Triton JIT warning for
`_fp8_mqa_topk_stream_kernel` if that exact stream-grouped shape was not covered
by warmup. Treat the second request at the same context length as the warm
measurement.

For the 128k benchmark, use an uncached single request with 128,000 prompt token
IDs and 128 output tokens, then compute throughput from Prometheus metric
deltas:

```text
server_prefill_tok_per_s = request_prefill_kv_computed_tokens_delta / request_prefill_time_seconds_delta
server_decode_tok_per_s = generation_tokens_delta / request_decode_time_seconds_delta
```

Do not use warmed benchmark-average input-rate math as the prefill metric when
prefix cache could be involved. Confirm `vllm:prefix_cache_hits_total` delta is
zero for the measured request.
