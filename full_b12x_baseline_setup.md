# DeepSeek V4 Flash GB10 Full B12x Baseline Recipe

This is the first full-parity recipe for the b12x DeepSeek V4 path. It is kept
separate from `baseline_setup.md`, which remains the rollback mHC-only baseline.

## Target

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Target: two GB10/DGX Spark nodes, TP=2, PP=1, expert parallel disabled
- Context: `max_model_len=262144`
- Scheduler shape: `max_num_batched_tokens=4096`, `max_num_seqs=6`
- Memory: `gpu_memory_utilization=0.85`
- KV cache: fp8
- Model dtype: bf16
- Spec decode: DeepSeek MTP with `num_speculative_tokens=2`
- B12x commit: `7580ef26d9d84d1c7c088a051aba733573a48752`
- Required b12x subsystems: compressed MLA, paged-MQA indexer, mHC, WO projection, MoE
- Comparison baseline: `1074.92` server prefill tok/s, `34.54` decode tok/s, `122.01s` TTFT

## Build

Build the same Dockerfile as the mHC baseline; full mode is selected by runtime
environment only.

```bash
set -euo pipefail

export VLLM_ROOT=/home/aidendle94/Documents/workspace/vllm
export BASE_IMAGE=sparkrun-vllm-ds4-gb10:w4a8-cutlass-swiglu-config-20260522T114518Z-cuda13.2-nccl2.30-vllm-openai-base
export FULL_B12X_REF=$(git -C "${VLLM_ROOT}" rev-parse --short HEAD)
export FULL_B12X_IMAGE=sparkrun-vllm-ds4-gb10:${FULL_B12X_REF}-full-b12x-256k-cuda13.2-nccl2.30-vllm-openai-base

git -C "${VLLM_ROOT}" diff --check

cd "${VLLM_ROOT}"
DOCKER_BUILDKIT=1 docker build \
  --file Dockerfile.baseline \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VLLM_BASELINE_COMMIT=$(git rev-parse HEAD)" \
  --tag "${FULL_B12X_IMAGE}" \
  "${VLLM_ROOT}"
```

## Launch Recipe

```yaml
recipe_version: "1"
name: DeepSeek V4 Flash GB10 full b12x baseline
description: DeepSeek V4 Flash full b12x parity run for dual DGX Spark TP=2, no EP/A2A, fp8 MLA KV cache, MTP=2, and 256K context.
runtime: vllm-distributed
model: deepseek-ai/DeepSeek-V4-Flash
container: sparkrun-vllm-ds4-gb10:<commit>-full-b12x-256k-cuda13.2-nccl2.30-vllm-openai-base
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
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP: "0"
  VLLM_USE_B12X_DEEPSEEK_V4: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_STRICT: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_MHC: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_INDEXER: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_COMPRESSED_MLA: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_WO_PROJECTION: "1"
  VLLM_USE_B12X_DEEPSEEK_V4_MOE: "1"
  VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: "0"
  FLASHINFER_DISABLE_VERSION_CHECK: "1"
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

Do not add `--enable-expert-parallel` to this recipe. The direct b12x MoE path
is intentionally no-EP for the first parity run.

## Validation

Required startup evidence:

```text
B12x DeepSeek V4 status: ... subsystem=compressed MLA requested=True strict=True
B12x DeepSeek V4 status: ... subsystem=paged MQA indexer requested=True strict=True
B12x DeepSeek V4 status: ... subsystem=mHC requested=True strict=True
B12x DeepSeek V4 status: ... subsystem=WO projection requested=True strict=True
B12x DeepSeek V4 status: ... subsystem=MoE requested=True strict=True
Using B12x direct W4A16 MoE path.
Using B12x rs-6 WO projection MXFP8 path.
```

Promotion checks:

- `/health` is healthy after the benchmark.
- `/v1/models` reports `max_model_len=262144`.
- Strict-mode status rows show all five b12x subsystems active.
- The 128K measured `/v1/completions` run computes `131072` prefill tokens with zero prefix hits.
- Report server prefill tok/s, decode tok/s, TTFT, computed prefill tokens, prefix hits, and status rows against the mHC baseline above.
