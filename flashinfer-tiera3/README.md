# flashinfer-tiera3 — flashinfer sampler overlay (`dsv4-tiera3:local`)

Layers two merged-upstream **flashinfer** fixes onto the dsv4-tiera2 image. Unlike
vLLM `csrc/` CUDA picks (which cannot take effect on the b12x *prebuilt-binary* base
— see `../DSV4_TIERA3.md`), flashinfer **JIT-compiles its kernels at runtime**, so a
flashinfer file overlay IS runtime-effective.

## TL;DR

```bash
# base: ../dsv4-tiera2-build/build.sh  -> dsv4-tiera2-src:local
./build.sh                            # -> dsv4-tiera3:local
# or overlay the running prod image directly:
BASE=dsv4-tiera2:local ./build.sh     # -> dsv4-tiera3:local
```

## What it overlays

| File | PR | Kind | Effect |
|---|---|---|---|
| `flashinfer/sampling.py` | [#3461](https://github.com/flashinfer-ai/flashinfer/pull/3461) | Python | `top_k_first` large-vocab small-k sampler fast path (2-4x at small batch; distribution-equivalent, PR TV ~0.01). Takes effect on import. |
| `flashinfer/data/include/flashinfer/sampling.cuh` | [#3624](https://github.com/flashinfer-ai/flashinfer/pull/3624) | CUDA JIT header | Fix shared-memory race (missing `__syncthreads`) + OOB token-id write in `SamplingFromLogitsKernel`. Recompiled by JIT on next sampling call (build busts the JIT cache). |

## ★ Version note (drift caught this round)

The source tree's `requirements/cuda.txt` pins `flashinfer-python==0.6.8.post1`, but
the **actual production base image ships flashinfer 0.6.12**. Both overlay files were
re-derived against the real **0.6.12** base files (extracted from the image), not the
source pin. The Dockerfile asserts `flashinfer == 0.6.12` as a pre-flight so a base
drift aborts the build loudly instead of mis-overlaying.

- `overlay/` — the two patched files at their exact site-packages-relative paths.
- `patches/` — reviewable `base(0.6.12) -> overlay` diffs (the source-of-record).

## Verification (performed)

- AST + `py_compile` of the patched `sampling.py`; the overlay diff vs the 0.6.12 base
  is exactly +114 lines (the three #3461 insertions, decorators untouched).
- In-image gate (Dockerfile): flashinfer == 0.6.12, both files replaced, JIT cache
  busted, `import vllm` + `import flashinfer` clean, #3461 symbols present, #3624 `.cuh`
  markers present.
- Deep runtime check in the built image: fast-path helpers callable with correct
  signatures; the `_top_k_first_fast_path_applicable` gate returns the expected
  True/False across boundary cases (big-vocab+small-k → True; k>256, vocab<65536,
  indices-set → False); both public samplers intact.
- **Full GPU serve smoke deferred:** the overlay touches only flashinfer's sampling
  path (not the DSv4 model / tool-calling / MTP / KV cache), so regression risk is
  structurally bounded; a live serve needs a GB10 node and contends with the running
  prod `dsv4-tiera2:local` for unified memory. Promotion + live gates are the
  operator's call — see `../DSV4_TIERA3.md`.
