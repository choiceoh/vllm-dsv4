# Experiments List

## 1. FlashInfer B12x W4A16 Expert Path for DeepSeek V4 on SM120

Status: proposed

Source:
- FlashInfer PR: https://github.com/flashinfer-ai/flashinfer/pull/3336
- Local FlashInfer checkout: `/home/aidendle94/Documents/workspace/flashinfer-latest`
- Local branch: `pr-3336-w4a16-rewrite`
- PR head inspected: `a52ad3a649ab`

### Hypothesis

DeepSeek V4 on SM120 may get faster expert execution by routing its current
FP4-weight/BF16-activation fallback through FlashInfer's new B12x W4A16 path
instead of the current vLLM SM120 W4A16 fallback path.

This is not a W4A4 experiment. It is a "make the W4A16 reality faster"
experiment.

### Why This Matters

The current DS4 SM120 bottleneck is the MoE expert path. DS4 has FP4 expert
weights, but SM120 backend support is incomplete enough that practical execution
often lands in W4A16-style behavior: 4-bit weights with BF16 activations.

FlashInfer PR #3336 specifically rewrites the SM120 W4A16 B12x path:
- Adds `b12x_fused_moe(..., quant_mode="w4a16")`.
- Adds `source_format="compressed_tensors"` handling.
- Replaces the prior W4A16 static/dynamic/micro split with a packed-route path.
- Uses a CuTe DSL persistent GEMM pipeline for FC1, activation, FC2, and output.
- Keeps route packing in Triton.

The important point for DS4 is that this targets the fallback mode we can
actually use sooner than native W4A4.

### Expected Benefit

Potential decode-side improvement from reducing W4A16 routing and small-batch
expert overhead on SM120.

Potential prefill improvement is less certain and must be measured. The path
still carries BF16 activation bandwidth and still has route-packing overhead.

### Non-Goals

- Do not treat this as true W4A4.
- Do not assume installing the FlashInfer PR changes vLLM behavior by itself.
- Do not launch a full DS4 server before a standalone expert-shape microbench
  proves the path is worth integrating.

### Integration Sketch

1. Build or install FlashInfer from `pr-3336-w4a16-rewrite`.
2. Write a standalone DS4-shape B12x W4A16 microbench using:
   - `flashinfer.b12x_fused_moe`
   - `quant_mode="w4a16"`
   - `source_format="compressed_tensors"`
   - DS4 expert dimensions, top-k, and representative decode/prefill token counts.
3. Compare against the current vLLM SM120 W4A16 expert backend.
4. Only if faster, add a vLLM expert wrapper that calls the B12x W4A16 API.
5. Re-run DS4 end-to-end decode and prefill benchmarks after backend integration.

### Acceptance Gate

Proceed to vLLM integration only if the standalone expert microbench beats the
current SM120 fallback at DS4-relevant decode shapes without adding new JIT
stutter to the hot path.

Minimum evidence before integration:
- Correctness check against a BF16/dequant reference.
- Warmed steady-state timing.
- First-call/JIT timing called out separately.
- Decode-shape comparison for small token counts.
- Prefill-shape comparison for larger routed-row counts.

### Known Risks

- The new FlashInfer path is not wired into vLLM's DS4 route today.
- Current vLLM FlashInfer CuTe DSL MoE wrapper calls the NVFP4/W4A4-oriented API,
  not `b12x_fused_moe(..., quant_mode="w4a16")`.
- Route packing is still Triton.
- Weight and scale layout compatibility needs explicit validation against DS4's
  loaded MXFP4/compressed-tensors representation.
- If the packed-route overhead dominates at `C=1`, this can still be slower.

## 2. SM12x Fused MQA Prefill Top-K for DeepSeek V4 Indexing

Status: prototype implemented, opt-in only

Source:
- Prototype kernel: `vllm/v1/attention/ops/deepseek_v4_ops/sm12x_mqa.py`
- Fallback gate: `vllm/v1/attention/ops/deepseek_v4_ops/sm12x_deep_gemm_fallbacks.py`
- Microbench: `benchmarks/benchmark_sm12x_mqa_topk.py`
- Focused test: `tests/v1/attention/test_sm120_deepgemm_fallbacks.py`

### Hypothesis

Large-prefill sparse-indexer work on SM12x can be sped up by replacing the
current chunked PyTorch MQA logits plus `torch.topk` path with a fused Triton
kernel that computes MQA scores and maintains exact row-wise top-k state.

The prototype does not materialize a full logits matrix and does not call
`torch.topk` in the fused path. It streams `topk`-wide KV tiles, merges each
tile into a persistent per-row top-k set, and writes final `int32` indices.

### Why This Matters

The 128k prefill profile showed the sparse attention indexer as a major
contributor, with `mbtopk` and PyTorch elementwise work visible in the hot path.
For large prompt chunks, the existing fallback spends substantial time in
materialized/chunked logits, top-k selection, gather/copy glue, and Python-level
launch loops.

This experiment targets that specific indexing slice, not the sparse MLA
attention accumulation itself.

### Expected Benefit

Standalone GB10 microbench results on synthetic DS4-shaped FP8 tensors:

- `rows=128, KV=32k, topk=2048`: `43.88 ms -> 26.55 ms`, about `1.65x`.
- `rows=128, KV=8k, topk=2048`: `10.22 ms -> 4.61 ms`, about `2.22x`.
- `rows=128, KV=8k, topk=512`: `8.23 ms -> 1.91 ms`, about `4.30x`.

This is an indexing-slice win, not an end-to-end prefill win. If indexing is
roughly 20-25% of the total prefill time, the expected end-to-end gain is
approximately 8-10%. It could be higher on profiles where indexing dominates.

### Clamp

The prototype is gated behind:

```bash
VLLM_SM12X_MQA_TOPK_TRITON=1
```

Even when enabled, it only runs for large-prefill-like shapes:

- `rows >= 64`
- `rows >= 128` when `topk=2048`
- `rows <= 256`
- `kv_tokens >= 8192`
- `topk in (512, 2048)`

The default clamps can be tuned with:

```bash
VLLM_SM12X_MQA_TOPK_TRITON_MIN_ROWS=64
VLLM_SM12X_MQA_TOPK_TRITON_MAX_ROWS=256
VLLM_SM12X_MQA_TOPK_TRITON_MIN_KV_TOKENS=8192
```

The clamp is required because small-row cases regressed. For example,
`rows=1, KV=8k, topk=2048` measured `0.92 ms -> 1.52 ms`, so decode-like
or tiny-prefill shapes should keep the existing fallback. The stricter
`topk=2048` row floor is required because `rows=64` was only break-even at
8K KV and showed threshold-level set differences at 32K KV. Very large chunks,
including broad 128k-row indexing chunks, also stay on the existing fallback
unless the max-row gate is explicitly relaxed after separate validation.

### Non-Goals

- Do not enable this unconditionally.
- Do not broaden this beyond FP8 gathered-prefill top-k without a separate
  validation pass.
- Do not treat the microbench speedup as full prefill speedup.
- Do not use this for FP4 Q/cache paths until separately implemented and tested.
- Do not replace the dense topk+SWA materialization experiment; that is a
  separate sparse MLA index-consumption problem.

### Integration Sketch

1. Keep the Triton prototype behind the current opt-in env gate.
2. Run standalone correctness against `torch.topk` for the representative
   `topk=512` and `topk=2048` shapes.
3. Run the microbench across `rows={1,32,64,128}`, `KV={8k,32k,128k}`, and
   `topk={512,2048}` to refine the clamp.
4. Launch DS4 only after standalone results remain favorable.
5. On full service, confirm server-side prefill throughput improves and that
   `mbtopk`/`aten::topk` disappear or shrink in the large-prefill profile.

### Acceptance Gate

Keep this only if all are true:

- Correctness matches the current path for large-prefill FP8 shapes.
- No regression for small-row shapes due to the clamp.
- Large-prefill microbench stays at least `1.5x` faster for target shapes.
- End-to-end 128k prefill improves measurably with zero prefix-cache hits.
- CUDA graph capture and warm serve behavior remain stable.

### Known Risks

- The current prototype is Triton, not a final DeepGEMM C++ kernel.
- `topk=2048` uses large per-row sort/merge state and can be slower at small
  row counts.
- Exact top-k tie ordering must be monitored; the focused synthetic tests passed,
  but broader model-path parity still needs full service validation. Candidate
  set equality is the intended contract; order-only differences are acceptable
  because sparse MLA consumes the selected indices as an unordered set.
- The path currently targets FP8 Q/KV only.
