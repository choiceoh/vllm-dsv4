# DSV4-tiera2 — DeepSeek-V4 (b12x) vLLM source branch

This branch (`dsv4-tiera2`) is the **source-level record** of the vLLM patch
stack running in the Deneb production image `dsv4-tiera2:local` (the DeepSeek-V4
main chat model on the GB10 / DGX Spark fleet). It exists so that future work —
porting to vLLM 0.23, adding more cherry-picks, rebasing the b12x base — can be
done as reviewable **pull requests** instead of opaque in-image string patches.

> **This is a Python-source + diff-management branch, not a from-scratch buildable
> tree.** The b12x MoE/MLA CUDA kernels are prebuilt binaries that live only in
> the Aiden base image, never in this source. See [Build model](#build-model).

---

## Base

- **Fork:** `choiceoh/vllm` (forked from `aidendle94/vllm`).
- **Branch base:** `DSV4` @ `eb99b8b` ("wip", 2026-05-24) — aidendle94's b12x
  DeepSeek-V4 source. vLLM `0.21.1rc1.dev339`. This carries the b12x MoE/MLA
  source: `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`,
  `vllm/v1/attention/backends/mla/b12x_integration.py`, the DSV4 attention
  adapter, W4A8 CUTLASS MoE, and `full_b12x_baseline_setup.md`.
- **Running image base commit:** the production image `dsv4-tiera2:local` was
  actually built from the Aiden b12x snapshot at vLLM commit
  `1967a5627bc3710b680bbec24ecb99aaddedf22b` (per `/opt/dsv4/provenance.json` in
  the image), which is a **slightly later snapshot than `DSV4` @ `eb99b8b`**.
  This matters: see [Source vs image divergence](#source-vs-image-divergence).

---

## Applied PRs (9) + leakfix

Each PR is exactly **one git commit** (`cherry-pick vllm#NNNNN: <title>` /
`cherry-pick deneb#2289 (leakfix)`). Commit order matches the runtime image's
layer order: base tier → leakfix → new tier.

### Base tier (5) — landed in image `dsv4-tiera:local` via a whole-file overlay

| PR | Title | Why (dsv4 relevance) | Apply |
|---|---|---|---|
| [#43733](https://github.com/vllm-project/vllm/pull/43733) | [Bugfix][DFlash] allocate the proper number of lookahead slots | dsv4 runs an MTP/DFlash draft; correct lookahead-slot count is load-bearing for spec-decode correctness | clean `git apply` |
| [#43445](https://github.com/vllm-project/vllm/pull/43445) | [Spec Decode] Allow causal DFlash | dsv4's DFlash/MTP draft reads these attention flags | clean `git apply` |
| [#42971](https://github.com/vllm-project/vllm/pull/42971) | Fix DFlash prefix cache corruption due to missing lookahead block | dsv4 runs DFlash with prefix caching (APC) | clean `git apply` (incl. upstream test) |
| [#44253](https://github.com/vllm-project/vllm/pull/44253) | [Bug Fix][MRv2][Spec Decode] Warmup & capture with different attention states for speculator prefill | dsv4 captures spec-decode (MTP) CUDA graphs | 3/4 files clean; `cudagraph_utils.py` from image (return-type rename hunk) |
| [#44420](https://github.com/vllm-project/vllm/pull/44420) | [feature] add index share feature for DSA MTP | dsv4 **is** a DSA + MTP model — this is the index-share path (image tag `dsv4-tiera:idxshare`) | 4/5 files clean; `deepseek_v2.py` from image (`__init__` drift) |

### Leakfix — `deneb#2289` (between base and new tier, = image `dsv4-tiera:local`)

Local Deneb fix (not an upstream vLLM PR), two parts, both stopping a KV-pool /
Automatic-Prefix-Cache (APC) leak that collapsed the prefix-cache hit rate:

1. `block_pool._maybe_evict_cached_block` — reset the block hash even when the
   hash was already absent from the map (a reused block keeping a stale
   `_block_hash` leaks pool usage).
2. `SlidingWindowMLAManager.cache_blocks` — protect only the stable aligned
   cache-hit prefix, not the transient tail that eagle/MTP recomputes each
   request (a fresh `block_id` every time → `~+1.5` protected blocks/req with no
   reuse benefit). Pure eviction-resistance optimization → cannot affect
   correctness, only hit rate.

### New tier (4) — landed in image `dsv4-tiera2:local` via anchored substitution

| PR | Title | Why (dsv4 relevance) | Apply |
|---|---|---|---|
| [#43961](https://github.com/vllm-project/vllm/pull/43961) | [Bugfix] Corrupted MLA + linear attention | dsv4 is MLA; without the `MLAAttentionSpec` gate, MLA blocks aren't reported once the KV cache fills → corrupt tail page | anchored substitution (×2 sites) |
| [#44821](https://github.com/vllm-project/vllm/pull/44821) | fix: prefix DeepSeek V4 MTP projections | dsv4 MTP `e_proj`/`h_proj` use fp8 quant; an empty `layer_name` breaks compressed-tensors matching | anchored substitution (path-adapted) |
| [#43991](https://github.com/vllm-project/vllm/pull/43991) | [MRv2] Use actual batch max_seq_len for attn metadata | dsv4's MTP draft (eagle/speculator) walks past valid block-table entries when handed `max_model_len` | anchored substitution (×6 anchors) |
| [#44603](https://github.com/vllm-project/vllm/pull/44603) | fix: pad dummy run query_start_loc | dsv4 (MLA + indexer) exercises the MLA-indexer dummy run during CUDA-graph capture; stale `query_start_loc` → `Assertion repeat >= 0 failed` | anchored substitution |

### Skipped

- [#43988](https://github.com/vllm-project/vllm/pull/43988) — **intentionally not
  applied.** The b12x base already handles the compressed reshape via a divergent
  code path, the PR's anchor is absent in this base, and the production config
  runs the exact trigger case without crashing. Recorded here so the skip is not
  mistaken for an omission.

---

## How the patches were applied (and how to re-derive)

Two mechanisms, mirroring how the runtime image was built:

1. **Upstream diff (`gh pr diff <N> -R vllm-project/vllm` → `git apply`).** Used
   for every base-tier file that applied cleanly against the `DSV4` base. The 5
   base-tier PRs are 0.23-era; most hunks still apply on `0.21.1rc1.dev339`.

2. **Anchored OLD/NEW substitution.** The 4 new-tier PRs + the leakfix are
   applied exactly as the running image applies them — the same
   `assert count == 1` anchored Python substitutions baked into the images
   (`dsv4-tiera:local:/tmp/apply_patch{,2}.py` for the leakfix;
   `~/tiera2_build/apply_patch_{43961,44821,43991,44603}.py` for the new tier).
   This guarantees the new-tier hunks are byte-identical to production. Two
   anchor-script files had to be **path-adapted** for the git source layout
   (the image installs DSV4 under a nested package — see below).

For the two base-tier files where the upstream diff would not land cleanly
(`cudagraph_utils.py` for #44253, `deepseek_v2.py` for #44420) the post-patch
file was taken straight from the running image. In both cases a line-by-line
audit confirmed the image file is `DSV4-base + that one PR` with **zero** foreign
changes, so the result is byte-identical to production for those files.

### Verification (source only — no build/runtime here)

- `ast.parse` over all 18 touched `.py` files — clean.
- New-tier files touched **only** by a single PR are byte-identical to
  `dsv4-tiera2:local` (`model_states/default.py`, `model_states/mamba_hybrid.py`
  — 0-line diff).
- Files touched by multiple PRs (`speculator.py`, `single_type_kv_cache_manager.py`,
  `gpu_model_runner.py`) match the image **modulo pre-existing snapshot drift**
  (see below) — the PR-specific lines themselves are byte-identical.

Build / runtime validation is the **next step** (see [Build model](#build-model)),
not done on this branch.

---

## Source vs image divergence

The `DSV4` git source (`eb99b8b`) and the image source (`1967a5627`) are two
**different snapshots** of the Aiden b12x tree. So beyond the 9 PRs, some files
differ for reasons unrelated to any patch here. Known divergences observed while
verifying:

- **deepseek_v4 package layout.** The image installs the DSV4 model under a
  nested package `vllm/models/deepseek_v4/{nvidia,amd}/mtp.py` (with a
  `current_platform.is_cuda()` NVIDIA/AMD fork). The `DSV4` git source keeps the
  **flat** module `vllm/model_executor/models/deepseek_v4_mtp.py` and has **no
  AMD variant**. #44821's anchor script (which patches both nvidia and amd in the
  image) is therefore retargeted here to the single flat git-source file.
- `gpu_model_runner.py` — the image source is ~110 lines longer (mamba buffer
  refactor `MambaCopyBuffers`→`MambaBuffers`, breakable-cudagraph imports). None
  of this is from our PRs.
- `speculator.py` — `init_attn_backend` returns a 3-tuple in `eb99b8b` vs a
  4-tuple in the image source.
- `single_type_kv_cache_manager.py` — one cosmetic quote-style line
  (`'eagle_extra_cache_blocks'` vs `"..."`).

These are expected and are **not** regressions introduced by this branch. They
exist because the runtime image is built on a later b12x snapshot than the `DSV4`
branch tip. A future task can rebase this branch onto the exact image snapshot
(or onto vLLM 0.23) to close the gap.

---

## Build model

**The b12x kernels are binaries, not source.** The b12x MoE/MLA CUDA kernels
(and the CUTLASS/DeepGEMM/FlashInfer artifacts) are prebuilt for `sm120/sm121`
and live only in the Aiden base image. This branch carries the **Python source**
that runs on top of them. A pure source build is therefore **not** possible.

The runtime image is built as an **overlay on the Aiden base image**, roughly:

```dockerfile
# Conceptual — the production build layers source onto the prebuilt b12x base.
FROM aidendle94/sparkrun-vllm-ds4-gb10:production-ready   # b12x binaries
# ... overlay this branch's vllm/ source over site-packages ...
# ... then apply the anchored new-tier substitutions (apply_patch_*.py) ...
```

The live `dsv4-tiera2:local` build chain on the fleet host is:

```
Aiden b12x base  →  base-tier overlay (5 PRs, whole-file COPY)  =  dsv4-tiera:local (+ leakfix apply_patch{,2}.py)
                 →  new-tier anchored substitution (4 PRs)       =  dsv4-tiera2:local
```

A Dockerfile build harness that consumes **this branch's source** (instead of
ad-hoc in-image patches) is the intended follow-up. The new-tier apply scripts
already exist at `~/tiera2_build/` on the fleet host and serve as the reference.

### Current production image

- **Image:** `dsv4-tiera2:local` (DeepSeek-V4 main chat model, GB10 fleet).
- **Serving:** `--gpu-memory-utilization 0.82`, MTP draft acceptance ~44%.
- The actual launcher / serve flags live on the fleet host, not in this repo.

---

## PR / merge references

| PR | Merged | Upstream merge commit |
|---|---|---|
| #44253 | 2026-06-03 | `91945b6e4ade` |
| #43733 | 2026-05-27 | `7fb9c0197a31` |
| #43445 | 2026-05-28 | `9202ea6fda05` |
| #42971 | 2026-06-02 | `0eeba5eec17e` |
| #44420 | 2026-06-07 | `32f34d393524` |
| #43961 | 2026-05-29 | `d2889722ff3a` |
| #44821 | 2026-06-10 | `4673ca1d7869` |
| #43991 | 2026-06-02 | `1edfd09ffd1f` |
| #44603 | 2026-06-05 | `d2f70da11625` |
| deneb#2289 | (local) | — leakfix, not an upstream PR |
