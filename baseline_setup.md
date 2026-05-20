# DeepSeek V4 Flash GB10 M64 Row-Tiled Logits Baseline Setup

This baseline captures the working two-node GB10 setup validated on
2026-05-19 and 2026-05-20. It is meant to be a stable starting image for
future agents so they can build one baseline image from this vLLM repository
and then layer experiments on top of that image.

## What This Baseline Contains

- vLLM branch: `production-baseline-20260515`
- vLLM source savepoint: the commit that contains this file and
  `Dockerfile.baseline`
- FlashInfer source: `https://github.com/flashinfer-ai/flashinfer.git`
- FlashInfer ref: `refs/pull/3336/head`
- FlashInfer commit: `a52ad3a649ab3716efe738be24e837566117d2b3`
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Runtime shape: TP=2, PP=1, EP enabled, MTP=2, fp8 KV
- Default context shape: `max_model_len=262144`,
  `max_num_batched_tokens=8192`, `max_num_seqs=1`
- Extended-context validation shape: `max_model_len=500001` for a
  500000-token prompt plus one output token
- MoE backend: `FLASHINFER_B12X_MXFP4_BF16`
- Static MLA/MQA path: `VLLM_SM12X_MQA_TOPK_TRITON=1`
- DeepSeek V4 thinking defaults are passed at launch with
  `--default-chat-template-kwargs '{"thinking":true,"enable_thinking":true,"reasoning_effort":"high"}'`
- MQA top-k baseline: row-tiled materialized logits enabled with
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED=1` and
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE=512`
- MQA logits tile experiment:
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_M=64`,
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_N=128`,
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_D=64`, and
  `VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_NUM_WARPS=4`
- FP8 Lightning Indexer path: uses the CUTeDSL fused-indexer implementation
  when `cutlass` is importable, with the existing Triton FP8 indexer kept as
  the fallback path.
- Stability fixes: exact small MTP decode CUDA graph capture sizes, bounded
  MTP uniform-decode warmup request counts, SWA decode-threshold alignment,
  prefix-cache block protection for partially computed prompts, and cached
  CUTeDSL availability probing.
- Persistent compile cache: `VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache`,
  `TRITON_CACHE_DIR=/cache/huggingface/triton-cache`,
  `TORCHINDUCTOR_CACHE_DIR=/cache/huggingface/torchinductor-cache`, and
  `TRITON_CACHE_AUTOTUNING=1`
- Combined prefill opts:
  `VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS=512`,
  `VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS=1`,
  `VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES=2`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H=8`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D=32`,
  `VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS=4`,
  `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=4096`, and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE=1024`
- Small-context clamp removed:
  `VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS=1` and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_MIN_TOKENS=1`
- Blocked sparse MLA prefill accumulator enabled in the baseline image and
  recipe:
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM=1`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE=1`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C=16`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS=8`,
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS=4`, and
  `VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES=2`

The source-level blocked accumulator knob remains default-off. This baseline
opts in through the Dockerfile and launch recipe after service-level validation.
The failed dynamic sparse-MLA candidate-cap experiment is intentionally absent
from this baseline.

The current M64 + stability + CUTeDSL warm measurements are:

| context | max output | measured prefill tok/s | measured decode tok/s | prefix hits | MTP draft / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 1 | 1149.52 | n/a | 0 | 0 / 0 |
| 32k | 1 | 1248.08 | n/a | 0 | 0 / 0 |
| 128k | 1 | 1092.09 | n/a | 0 | 0 / 0 |
| 128k | 64 | 1120.66 | 35.80 | 0 | 44 / 42 |

The small-context clamp removal was validated on 2026-05-20 with warm
single-token requests against the same baseline image:

| context | clamped prefill tok/s | unclamped prefill tok/s | delta |
| ---: | ---: | ---: | ---: |
| 1k | 1080.74 | 1152.94 | +6.7% |
| 4k | 1245.84 | 1353.26 | +8.6% |
| 8k | 1197.05 | 1313.18 | +9.7% |
| 16k | 1185.94 | 1285.56 | +8.4% |
| 32k | 1119.89 | 1255.07 | +12.1% |

The tested pre-commit image was:

```text
sparkrun-vllm-ds4-gb10:53511e9c9-m64-stab-cutedsl-nooverlap-dirty-20260520T055057Z-cuda13.2-nccl2.30-vllm-openai-base
```

The rejected C128A gather/indexer overlap candidate is intentionally absent
from this baseline. It failed during startup with `NameError: attn_metadata is
not defined`.

The prior M32 q4096 row-tiled logits warm measurements were:

| context | max output | measured prefill tok/s | measured decode tok/s | prefix hits |
| ---: | ---: | ---: | ---: | ---: |
| 1k | 1 | 1131.00 | n/a | 0 |
| 32k | 1 | 1230.11 | n/a | 0 |
| 64k | 1 | 1182.19 | n/a | 0 |
| 128k | 1 | 1062.85 | n/a | 0 |
| 32k | 64 | 1153.17 | 41.35 | 0 |
| 128k | 64 | 1051.81 | 36.34 | 0 |

The promoted M32 logits-tile baseline preserved correctness on GSM8K and
validated the 500k context path:

| check | measured result | artifact |
| --- | ---: | --- |
| GSM8K 5-shot, limit 200 | 0.965 flexible EM, 0.955 strict EM | `/home/aidendle94/Documents/workspace/experiment_runs/logits_m32_gsm8k_20260519T233515Z` |
| 500000-token prompt, `max_tokens=1`, `max_model_len=500001` | 668.129 prefill tok/s, prefix hits 0 | `/home/aidendle94/Documents/workspace/experiment_runs/logits_m32_max500001_bench500k_20260520T002558Z` |

Exact `max_model_len=500000` rejected the 500000-token prompt with
`max_tokens=1` because the requested total context was 500001. Use 500001 for
that validation shape.

The prior row512/q2048 row-tiled logits warm measurements were:

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

This experimental stack promotes row512/q4096 plus blocked sparse MLA
accumulation plus row-tiled
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
/home/aidendle94/Documents/workspace/experiment_runs/blocked_tune_q4096_20260519T183400Z/report.md
/home/aidendle94/Documents/workspace/experiment_runs/logits_m32_gsm8k_20260519T233515Z
/home/aidendle94/Documents/workspace/experiment_runs/logits_m32_max500001_bench500k_20260520T002558Z/report.md
/home/aidendle94/Documents/workspace/experiment_runs/m64_stab_cutedsl_nooverlap_20260520T060207Z/report.md
```

## Build The Baseline Image

Build from the vLLM repo root. The Dockerfile fetches the pinned FlashInfer PR
source itself, so a GitHub checkout of this vLLM repo is sufficient as the build
context.

The only external image prerequisite is `BASE_IMAGE`: it should be the
CUDA 13.2 / NCCL 2.30 / vLLM OpenAI base image used on the GB10 cluster. Push
that base image to a registry or replace `BASE_IMAGE` with an equivalent image
before building on a different machine.

```bash
set -euo pipefail

export VLLM_ROOT=/home/aidendle94/Documents/workspace/vllm
export BASE_IMAGE=sparkrun-vllm-ds4-gb10:ae353d502-static-mla-dirty-20260518T000139Z-cuda13.2-nccl2.30-vllm-openai-base
export BASELINE_REF=$(git -C "${VLLM_ROOT}" rev-parse --short HEAD)
export BASELINE_IMAGE=sparkrun-vllm-ds4-gb10:${BASELINE_REF}-m64-rowtiled-logits-baseline-cuda13.2-nccl2.30-vllm-openai-base

git -C "${VLLM_ROOT}" diff --check

cd "${VLLM_ROOT}"
DOCKER_BUILDKIT=1 docker build \
  --file Dockerfile.baseline \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${BASELINE_IMAGE}" \
  "${VLLM_ROOT}"
```

The build-time assertion checks that vLLM can see the FlashInfer B12x W4A16
entrypoints. The Dockerfile pins FlashInfer to PR `3336` at commit
`a52ad3a649ab3716efe738be24e837566117d2b3`; it does not depend on
`/home/aidendle94/Documents/workspace/flashinfer-latest`. The Dockerfile also
bakes the persistent compile-cache path, row-tiled logits top-k env defaults,
M64 logits tile defaults, row512/q4096 runtime env defaults, H8/D32/W4 stream
fallback env defaults, and blocked sparse MLA accumulator env defaults listed
in the recipe below, so future overlay images inherit this baseline even when
the launch recipe does not restate every knob.

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
name: DeepSeek V4 Flash GB10 M64 row-tiled logits prefill baseline
description: DeepSeek V4 Flash GB10 baseline with FlashInfer B12x W4A16, static SM12x MQA top-k, row-tiled materialized logits top-k, M64 logits tiling, blocked sparse MLA prefill accumulation, combined prefill opts, MTP=2, and 262k default context.
runtime: vllm-distributed
model: deepseek-ai/DeepSeek-V4-Flash
container: sparkrun-vllm-ds4-gb10:<commit>-m64-rowtiled-logits-baseline-cuda13.2-nccl2.30-vllm-openai-base
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
  VLLM_CACHE_ROOT: /cache/huggingface/vllm-cache
  TRITON_CACHE_DIR: /cache/huggingface/triton-cache
  TRITON_CACHE_AUTOTUNING: "1"
  TORCHINDUCTOR_CACHE_DIR: /cache/huggingface/torchinductor-cache
  VLLM_ENFORCE_STRICT_TOOL_CALLING: "1"
  VLLM_USE_FLASHINFER_MOE_B12X_W4A16: "1"
  VLLM_SM12X_MQA_TOPK_TRITON: "1"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED: "1"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE: "512"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_M: "64"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_N: "128"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_D: "64"
  VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_NUM_WARPS: "4"
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS: "64"
  VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS: "512"
  VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS: "1"
  VLLM_SM12X_MQA_TOPK_TRITON_STREAM_K_TILES: "2"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H: "8"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D: "32"
  VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS: "4"
  VLLM_TRITON_MLA_SPARSE: "1"
  VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: "4096"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_SIZE: "1024"
  VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_MIN_TOKENS: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE: "1"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C: "16"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS: "8"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS: "4"
  VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES: "2"
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
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --tokenizer-mode deepseek_v4 \
    --default-chat-template-kwargs '{"thinking":true,"enable_thinking":true,"reasoning_effort":"high"}' \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
```

Launch it:

```bash
cd /home/aidendle94/Documents/workspace/vllm-ds4-sm120-harness
sparkrun run /tmp/deepseek-v4-flash-gb10-m64-rowtiled-logits-baseline.yaml --no-follow
```

For 500k validation, use the same image and recipe but set
`max_model_len: 500001`. A 500000-token prompt with one generated token needs a
500001 total context budget.

If reusing an older checked harness recipe, make sure it includes the
combined-prefill and blocked-accumulator env vars above, uses `MAX_ROWS=512`,
`MIN_KV_TOKENS=1`, `PREFILL_TOPK_CHUNK_MIN_TOKENS=1`,
`QUERY_CHUNK_SIZE=4096`, `PREFILL_BLOCK_C=16`, `PREFILL_BLOCK_HEADS=8`,
`PREFILL_BLOCK_WARPS=4`, `PREFILL_BLOCK_STAGES=2`, and
the row-tiled logits, M64 logits tile, and H8/D32/W4 fallback topk512 env vars,
and removes any `--profiler-config` unless you are intentionally taking a
profile.
The old static-MLA profile recipe is not the clean throughput baseline by
itself.

```bash
rg "LOGITS_ROW|LOGITS_BLOCK|MAX_ROWS|STREAM_K_TILES|TOPK512|QUERY_CHUNK|PREFILL_TOPK_CHUNK|BLOCKED_ACCUM|BLOCK_C|BLOCK_HEADS|profiler-config" sparkrun/*.yaml
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
  "default-chat-template|MIN_KV_TOKENS|TOPK_CHUNK_MIN|BLOCKED_ACCUM|PREFILL_BLOCK_C|PREFILL_BLOCK_HEADS|PREFILL_BLOCK_WARPS|PREFILL_BLOCK_STAGES|QUERY_CHUNK_SIZE|LOGITS_ROW|LOGITS_BLOCK|MAX_ROWS|TOPK512"
```

Expected recipe evidence:

```text
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM: '1'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCKED_ACCUM_FP32_VALUE: '1'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_C: '16'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_HEADS: '8'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_WARPS: '4'
VLLM_TRITON_MLA_SPARSE_PREFILL_BLOCK_STAGES: '2'
VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE: '4096'
VLLM_TRITON_MLA_SPARSE_PREFILL_TOPK_CHUNK_MIN_TOKENS: '1'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILED: '1'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_ROW_TILE: '512'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_M: '64'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_N: '128'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_BLOCK_D: '64'
VLLM_SM12X_MQA_TOPK_TRITON_LOGITS_NUM_WARPS: '4'
VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS: '512'
VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS: '1'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_H: '8'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_BLOCK_D: '32'
VLLM_SM12X_MQA_TOPK_TRITON_TOPK512_NUM_WARPS: '4'
```

Startup warmup should cover the row-tiled logits, sparse MLA metadata, MTP
decode prep, and FlashInfer route-pack kernels before the JIT monitor is
activated. If a new shape still appears, it should be written under
`/cache/huggingface/vllm-cache` or `/cache/huggingface/triton-cache` so a
container restart on the same image and host can reuse it instead of
recompiling on the first user request.

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
