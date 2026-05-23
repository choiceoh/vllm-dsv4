# DeepSeek V4 Flash GB10 rs-6 B12x mHC Baseline

This is the repo-local rebuild path for the validated baseline promoted on
2026-05-23. The goal is that a future checkout of this vLLM branch can recreate
the baseline image without a separate overlay directory.

## Baseline Contents

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Target: two GB10/DGX Spark nodes, TP=2, PP=1, EP enabled
- Context: `max_model_len=262144`
- Scheduler shape: `max_num_batched_tokens=4096`, `max_num_seqs=6`
- Memory: `gpu_memory_utilization=0.85`
- KV cache: fp8
- Model dtype: bf16
- Spec decode: DeepSeek MTP with `num_speculative_tokens=2`
- Chat defaults: `{"enable_thinking":true,"reasoning_effort":"high"}`
- Attention path: vLLM sparse MLA/MQA plus rs-6 B12x mHC pre/post hooks
- B12x commit: `afbbf71dc47931988646399587c3af1c9be3ee18`
- B12x DeepSeek V4 mHC: enabled
- B12x compressed MLA and paged-MQA indexer: intentionally disabled by default
- MoE path: FlashInfer CUTLASS MXFP4/MXFP8 W4A8
- Old experimental `VLLM_USE_FLASHINFER_MOE_B12X_W4A8` route: intentionally absent

The baseline Dockerfile imports the vLLM source files directly from this repo
and installs the pinned B12x package during the image build. It does not depend
on `overlays/ds4-rs6-b12x-mhc`.

## Build

The only image prerequisite is the W4A8 CUTLASS vLLM OpenAI base image. The
validated local base was:

```text
sparkrun-vllm-ds4-gb10:w4a8-cutlass-swiglu-config-20260522T114518Z-cuda13.2-nccl2.30-vllm-openai-base
```

Build from the vLLM repo root:

```bash
set -euo pipefail

export VLLM_ROOT=/home/aidendle94/Documents/workspace/vllm
export BASE_IMAGE=sparkrun-vllm-ds4-gb10:w4a8-cutlass-swiglu-config-20260522T114518Z-cuda13.2-nccl2.30-vllm-openai-base
export BASELINE_REF=$(git -C "${VLLM_ROOT}" rev-parse --short HEAD)
export BASELINE_IMAGE=sparkrun-vllm-ds4-gb10:${BASELINE_REF}-rs6-b12x-mhc-256k-cuda13.2-nccl2.30-vllm-openai-base

git -C "${VLLM_ROOT}" diff --check

cd "${VLLM_ROOT}"
DOCKER_BUILDKIT=1 docker build \
  --file Dockerfile.baseline \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VLLM_BASELINE_COMMIT=$(git rev-parse HEAD)" \
  --tag "${BASELINE_IMAGE}" \
  "${VLLM_ROOT}"
```

The build assertion checks that vLLM can import the repo-local DeepSeek V4 B12x
integration module, the pinned `b12x.integration` package, the B12x mHC
entrypoints, and the FlashInfer CUTLASS/B12x MoE entrypoints.

## Launch Recipe

Use the image built above as `container`.

```yaml
recipe_version: "1"
name: DeepSeek V4 Flash GB10 rs-6 B12x mHC baseline
description: Validated DeepSeek V4 Flash GB10 baseline for dual DGX Spark TP=2 with rs-6 B12x mHC, CUTLASS W4A8 MoE, fp8 MLA KV cache, MTP=2, high-effort thinking defaults, and 256K context.
runtime: vllm-distributed
model: deepseek-ai/DeepSeek-V4-Flash
container: sparkrun-vllm-ds4-gb10:<commit>-rs6-b12x-mhc-256k-cuda13.2-nccl2.30-vllm-openai-base
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
  max_num_batched_tokens: 4096
  max_num_seqs: 6
  block_size: 256
  kv_cache_dtype: fp8
  served_model_name: deepseek-v4-flash

env:
  TORCH_CUDA_ARCH_LIST: 12.1a
  VLLM_ENFORCE_STRICT_TOOL_CALLING: "1"
  VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP: "0"
  VLLM_USE_B12X_DEEPSEEK_V4: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_MHC: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_INDEXER: "0"
  VLLM_USE_B12X_DEEPSEEK_V4_COMPRESSED_MLA: "0"
  VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: "1"
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
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  VLLM_TRITON_MLA_SPARSE: "1"
  VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: "2048"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE: "1024"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_MIN_TOKENS: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C: "16"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS: "8"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS: "4"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES: "2"
  VLLM_SM12X_DIRECT_FP8DS_PREFILL: "0"
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
  VLLM_NCCL_SO_PATH: /usr/lib/aarch64-linux-gnu/libnccl.so.2
  FLASHINFER_DISABLE_VERSION_CHECK: "1"
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
    --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    --load-format safetensors
```

## Validation

Health checks:

```bash
curl -fsS --max-time 10 http://192.168.50.29:8000/health
curl -fsS --max-time 10 http://192.168.50.29:8000/v1/models
```

Expected runtime evidence:

```text
max_model_len=262144
gpu_memory_utilization=0.85
max_num_seqs=6
kv_cache_dtype=fp8
Using B12x mHC path
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend
```

Validation suite for promotion:

- 128K completion smoke
- 256K completion smoke
- 15-request tool-call harness
- Haystack at 1K and 128K
- GSM8K 300
- C=4 request with 16K context
