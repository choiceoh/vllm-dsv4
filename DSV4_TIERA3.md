# DSV4-tiera3 — kernel-pick stack on top of dsv4-tiera2

This branch (`dsv4-tiera3`) extends [`dsv4-tiera2`](./DSV4_TIERA2.md) with an
additional round of kernel picks (bug fixes + perf), evaluated against the hard
constraint of how the production image is actually built.

> **Read [DSV4_TIERA2.md](./DSV4_TIERA2.md) first.** It establishes the decisive
> build model that governs every decision here.

---

## The build model decides everything (recap + its consequence)

The production image `dsv4-tiera2:local` is a **Python-source overlay on the Aiden
b12x *prebuilt-binary* base image**. The b12x MoE/MLA CUDA kernels (CUTLASS W4A8,
DeepGEMM, FlashInfer cubins for sm120/sm121) are prebuilt binaries that live only
in the base image and cannot be compiled from any public source tree. The build is
`FROM aidendle94/...:production-ready` + `git apply` a **Python-only** patch +
`py_compile`. **`csrc/` is never recompiled.**

Two hard consequences for picks:

1. **vLLM `csrc/` CUDA picks cannot take runtime effect in the overlay image.**
   Editing `csrc/*.cu` changes the source tree but not the inherited prebuilt
   `.so`/cubins. To make such a fix live you need a *from-source* base that ships
   the b12x kernels — which does not exist publicly. So `csrc` CUDA picks here are
   committed as the **correct source-level record** (one-commit-per-PR, ready for a
   future from-source base), with **no effect on the deployed overlay image**.

2. **flashinfer picks DO take runtime effect.** The production base image ships
   **flashinfer 0.6.12** (★ note: the source tree's `requirements/cuda.txt` pins
   `0.6.8.post1`, but the actual deployed base image is 0.6.12 — the overlays were
   re-derived against the real 0.6.12 base files, which this round caught). The
   flashinfer wheel is `py3-none-any` and ships **zero compiled kernels** — it
   JIT-compiles every CUDA kernel at runtime (NVRTC/nvcc) from in-wheel `.cu`/`.cuh`
   source. So overlaying `flashinfer/sampling.py` (pure Python) or
   `flashinfer/.../sampling.cuh` (JIT header, + a JIT-cache bust) is a valid,
   runtime-effective mechanism on a GPU host.

This split is the spine of the tiera3 plan: **the deliverable runtime change is a
flashinfer overlay; the vLLM csrc fixes are source-record only.**

---

## Pick decisions

| Pick | Type | Lang | Verdict | Where it lands |
|---|---|---|---|---|
| flashinfer **#3461** | perf (sampler) | Python | **APPLIED — runtime-effective** | `flashinfer-tiera3/` overlay → `dsv4-tiera3:local` |
| flashinfer **#3624** | bug (race/OOB) | CUDA `.cuh` (JIT) | **APPLIED — runtime-effective via JIT** | `flashinfer-tiera3/` overlay → `dsv4-tiera3:local` |
| vllm **#42379** | bug (RMSNorm fp32 cast) | CUDA `.cu` | **SOURCE-PATCHED** (no overlay effect) | git commit on `csrc/layernorm_*.cu` |
| vllm **#45255** | bug (gridDim.y 65535) | CUDA `.cu` | **SOURCE-PATCHED** (no overlay effect) | git commit on `csrc/libtorch_stable/.../fp8/per_token_group_quant.cu` |
| vllm **#42169** | bug (topk stride) | CUDA | **SKIP — already fixed** | merge `986edc85` is an ancestor of HEAD |
| **b12x ×7** (cb98da162, c7089a418, 8f7fe8792, edf8d3dba, 1e9b2693e, d62eaee9f, 0ff2847b0) | perf | Python (CuteDSL) | **SKIP — ABI break** | see below |
| vllm **#43162** | perf (fuse q-pad) | CUDA+Py | **SKIP** | csrc no-recompile + Py at reorg paths absent in fork |
| vllm **#43554** | perf/refactor (rm NormGateLinear) | CUDA+Py | **SKIP** | deletion refactor, not a fix; csrc no-recompile; Py model paths absent |
| vllm **#44173** | perf (silu+quant) | CUDA only | **SKIP** | pure csrc → no overlay effect |
| vllm **#43014** | perf (moe permute) | CUDA+Py | **SKIP** | Py overlay inert without the new csrc bindings (no recompile) |

---

## The two applied flashinfer picks (deliverable: `dsv4-tiera3:local`)

Harness: **`flashinfer-tiera3/`** (`./build.sh` → `dsv4-tiera3:local`, FROM
`dsv4-tiera2-src:local`; overlays two files, inherits the b12x stack unchanged).

### #3461 — top_k_first large-vocab small-k sampler fast path (`sampling.py`)
- **Merged** upstream (`49cb250f`). Pure Python (+114 / −0 in `flashinfer/sampling.py`).
- Adds a fast path: when `indices is None`, `top_k` is a scalar in `(0, 256]`, and
  vocab `>= 65536`, select top-k with the parallel radix kernel then run top-p over
  only the k survivors. Distribution-equivalent to the masked full-vocab path
  (PR-validated TV ~0.01); 2-4x faster at small batch on large vocab.
- **Applicability proven against the real base flashinfer 0.6.12** (extracted from
  the production-ready image, `py3-none-any`): not present in 0.6.12, and both
  dependencies exist — `topk.top_k(input, k, sorted=False, deterministic=False, ...)`
  (0.6.12 adds backward-compatible `tie_break`/`dsa_graph_safe` defaults) and
  `top_p_sampling_from_probs(..., return_valid=False)`. No new kernel symbol; the
  radix top-k JIT-builds from in-wheel `topk.cu`.
- **★ Decorator note:** 0.6.12 already carries the `@flashinfer_api(trace=...)`
  decorators (unlike the 0.6.8.post1 source pin, where the `trace=` infra is absent).
  The overlay only **inserts** the fast-path blocks and leaves the decorators intact,
  so there is no NameError risk. The overlay diff is exactly the three functional
  insertions (helper block + two call-site guards), +114 lines, decorators untouched.
  (Re-deriving against the real base was necessary: building against 0.6.8.post1 would
  have shipped a stale-context file — this round caught the requirements-vs-image drift.)

### #3624 — SamplingFromLogitsKernel race + OOB token-id fix (`sampling.cuh`)
- **Merged** upstream (`78d0a06b`). 6-line change to the JIT header
  `flashinfer/data/include/flashinfer/sampling.cuh` (wheel path; PR path is
  `include/flashinfer/sampling.cuh`).
- Guards an out-of-range `token_idx` (writes `.index = 0` instead of OOB) and adds a
  `__syncthreads()` after the `BlockReduce` to close a shared-memory race on
  `temp_storage` reuse across the loop.
- **Runtime mechanism:** flashinfer JIT-compiles this kernel; the overlay replaces
  the header and busts the JIT cache so `SamplingFromLogitsKernel` recompiles on the
  next logits-sampling call. Requires the CUDA toolchain JIT uses (the GPU image has
  it). The patch context (`SamplingFromLogitsKernel` + the patched `cur_data[j].index`
  line) exists verbatim at the pin.

Both overlay files were produced by applying the upstream PRs to the **pinned wheel**
copies and verifying byte-for-byte that the diff is exactly the PR (minus the #3461
decorator change). Reviewable upstream patches are kept in
`flashinfer-tiera3/patches/`.

---

## The two source-patched vLLM csrc bug fixes (record only — no overlay effect)

These are **real** bugs present in the fork source. They are committed so the source
tree is correct and a future from-source base can ship them, but they **do not change
the deployed overlay image** (csrc is not recompiled there).

### #42379 — RMSNorm multiply in weight's native dtype (`csrc/layernorm_*.cu`)
- Real regression (DSV4 rebase #40860 reintroduced an fp32 weight upcast). The
  residual-add RMSNorm path did `x * s_variance * (float)w`, diverging from the
  unfused composite at quantization tie boundaries. Fix: `(scalar_t)(x*s_variance) * w`.
- All **6** affected kernels in the fork carried the buggy pattern and are fixed
  (`rms_norm_kernel`, vec+scalar `fused_add_rms_norm_kernel`, and the three fp8-quant
  variants in `layernorm_quant_kernels.cu`).
- Applied from `gh pr diff 42379` with a **path rewrite only** (upstream
  `csrc/libtorch_stable/layernorm_*.cu` → this fork's pre-migration `csrc/layernorm_*.cu`);
  `git apply --check` clean after the rename.

### #45255 — per_token_group_quant gridDim.y 65535 overflow (`csrc/libtorch_stable/.../fp8/per_token_group_quant.cu`)
- Real bug, same path as upstream. The packed FP8 per-token-group-quant kernel maps
  the mn (token/row) dim onto `blockIdx.y`, but CUDA caps grid.y/grid.z at 65535
  (grid.x at 2^31−1); the host guard wrongly checked `INT32_MAX` on both dims. With
  >65535 rows the launch fails. Fix: swap axes (rows→grid.x, sf_k→grid.y), rename
  `blocks_x/blocks_y`→`sf_k_blocks/row_blocks`, cap `sf_k_blocks <= 65535`.
- **Manually ported** (not a clean apply): upstream modifies a dual launch path
  (`#if cudaLaunchConfig_t / #else dim3 grid`) but this fork's snapshot has only the
  single `dim3 grid` macro, and the kernel hunk drifted (~L287 here vs ~L301
  upstream). The three logical changes were applied to the fork's single-launch
  structure. The int8 variant has no packed_register_kernel (unaffected; matches the
  upstream fp8-only scope). The upstream regression test (mn=65537) applied cleanly.

### #42169 — already in the fork (no-op)
The fork's merge commit *is* PR #42169's merge (`986edc85`, an ancestor of HEAD); the
topk stride fix (`logits.stride(0)`) is already present at both sites. Nothing to do.

---

## Why ALL seven b12x picks are skipped (the load-bearing skip)

The named b12x commits live in `lukealonso/b12x`. The production base pins
`B12X_COMMIT=afbbf71dc47931988646399587c3af1c9be3ee18`; the seven picks sit
**90–191 commits ahead** of that pin (the newest, `0ff2847b0`, is 5 behind master).
b12x is overwhelmingly Python (CuteDSL/Triton, JIT-compiled) — so in principle the
picks *are* the right "language" for an overlay. They are skipped for two concrete,
verified reasons:

1. **Individual cherry-pick is impossible.** Every pick fails with `modify/delete`:
   the files they touch (`b12x/gemm/block_fp8_linear.py`, `gemm/dense.py` rewrites,
   the entire `attention/paged/` tree, `attention/indexer/fused_indexer.py`) **do not
   exist at the pin** — they were created by intermediate commits in the 90–191-commit
   chain. A pick can only land by dragging that whole chain.

2. **Bumping `B12X_COMMIT` to pull the chain breaks the vLLM↔b12x integration ABI.**
   The only coherent way to get the picks is to advance the pin (all seven are
   ancestors of master). But the b12x→vLLM API changed incompatibly across that range.
   vLLM's `b12x_integration.py` (a COPY'd overlay file in the prod image) calls these
   with the **old `workspace=` signature**; the bump target moved to a `binding=`-based
   API:
   - `compressed_mla_decode_forward`: pin requires `workspace=B12XAttentionWorkspace`;
     target **removed `workspace`**, added `binding=None`/`backend=None`.
   - `b12x_mhc_pre`: pin has `workspace=`; target **removed it**, added
     `binding`/`norm_weight`/`norm_eps`.
   - `b12x_mhc_post`: pin has `workspace=` and args `post`/`comb`; target **removed
     `workspace`**, renamed to `prev_post`/`prev_comb`.
   Pulling the chain therefore requires a **matching rewrite of vLLM's
   `b12x_integration.py`** — which is exactly the *"b12x v0.23.0 전체흡수 (integration
   API 깨짐 재적응)"* large task that is **out of scope** for this round. (Several picks
   are additionally on disabled code paths: `VLLM_USE_B12X_DEEPSEEK_V4_INDEXER=0` and
   `..._COMPRESSED_MLA=0` gate off the fused indexer / paged-MSA work.)

Forcing the b12x picks would break the production model. This is a safety skip, not an
omission — see the [model-roles dogma](../.claude/rules) "안전 > 완전".

---

## Why the four vLLM perf picks are skipped

All four touch `csrc/` (so they cannot take effect via the overlay build), and their
Python halves either target the upstream DSv4 *reorg* paths
(`vllm/models/deepseek_v4/...`) that are **absent in this fork** (it keeps the classic
`vllm/model_executor/...` layout), or are **inert without the matching recompiled CUDA
bindings**:

- **#43162** (fuse q-pad): CUDA kernel signature change + Python at reorg paths; the
  fork calls the older 4-arg kernel binding. Needs a full manual port of both halves.
- **#43554** (remove NormGateLinear): a deletion/refactor, not a correctness fix;
  removes a CUDA kernel path + rewires DSv4 model code at absent reorg paths.
- **#44173** (silu+quant rewrite): pure CUDA, single file — overlay can't recompile it.
- **#43014** (moe permute): the only perf PR whose paths line up, but its Python parts
  call **new** csrc bindings (`moe_ops.h`/`torch_bindings.cpp`) that only exist after a
  CUDA recompile — a Python-only overlay would call unregistered ops.

These remain candidates for a from-source base; none is a correctness bug.

---

## Build & verification

### Deliverable image
```bash
# 1) Reproduce the dsv4-tiera2 source base (if not already built):
cd dsv4-tiera2-build && ./build.sh        # -> dsv4-tiera2-src:local
# 2) Layer the flashinfer overlay:
cd ../flashinfer-tiera3 && ./build.sh     # -> dsv4-tiera3:local
#    (to overlay the running prod image directly: BASE=dsv4-tiera2:local ./build.sh)
```
The overlay build runs an in-image gate: flashinfer == 0.6.12, both overlay
files present + replaced, JIT cache busted, `import vllm`/`import flashinfer` clean,
`#3461` fast-path symbols present, `#3624` `.cuh` markers present.

### Source-record csrc fixes
The #42379/#45255 commits are verified by source inspection (before/after at every
site) and AST/syntax checks. They are **not** built here (no from-source base); the
#45255 regression test (`tests/kernels/quantization/test_per_token_group_quant.py
::test_per_token_group_quant_fp8_packed_large_mn`) requires a CUDA GPU + a from-source
build to actually exercise the kernel.

### What full live validation needs (and why it's deferred)
The sampler overlay is on flashinfer's sampling path only — it does **not** touch the
DeepSeek-V4 model, tool-calling, MTP draft, or KV cache. So tool-calling / decode /
MTP-accept regression risk from the overlay is structurally near-zero. A full live
serve (tool-calling regression, decode tok/s, MTP accept ≥44%, Korean quality) should
still be run before promotion, but it requires booting `dsv4-tiera3:local` on a GB10
node — which contends for unified memory with the running prod `dsv4-tiera2:local`.
Promotion is the operator's call (this branch only builds + verifies the overlay).

---

## Promotion (operator decision)
`dsv4-tiera3:local` is a drop-in for `dsv4-tiera2:local` (same b12x stack, two patched
flashinfer files). To promote: point the fleet launcher at `dsv4-tiera3:local`, boot on
a GB10 node, and run the live gates above. The vLLM csrc source fixes do not affect the
overlay image and need no promotion action.
