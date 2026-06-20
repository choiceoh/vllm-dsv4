# syntax=docker/dockerfile:1
# =============================================================================
# dsv4-tiera3 — flashinfer sampler overlay on top of the dsv4-tiera2 image.
#
# WHY A SEPARATE OVERLAY (and why this one DOES take runtime effect)
# -----------------------------------------------------------------
# The dsv4-tiera2 production lineage is a Python-source overlay on the Aiden
# b12x *prebuilt-binary* base. That means vLLM `csrc/` CUDA changes cannot take
# effect (csrc is never recompiled in the build) — see DSV4_TIERA3.md. flashinfer
# is different: the base image ships flashinfer 0.6.12 (NOTE: the source tree's
# requirements/cuda.txt pins 0.6.8.post1, but the actual production base image is
# 0.6.12 — these overlays were re-derived against the real 0.6.12 base files). The
# flashinfer wheel is `py3-none-any` and ships ZERO compiled kernels — it
# JIT-compiles every CUDA kernel at runtime (NVRTC/nvcc) from in-wheel `.cu`/`.cuh`
# source. So overlaying a flashinfer file IS a valid, runtime-effective mechanism:
#   * sampling.py   — pure Python, takes effect on import (no recompile at all).
#   * sampling.cuh  — JIT header; the host's nvcc recompiles SamplingFromLogitsKernel
#                     on the next sampling call once the JIT cache is busted.
#
# WHAT THIS OVERLAYS (two merged-upstream flashinfer fixes, verified at the pin)
#   * PR #3461 (sampling.py): top_k_first large-vocab small-k fast path
#       (parallel radix top-k, then top-p over only the k survivors). Hot-path
#       sampler speedup (2-4x for large-vocab small-k), distribution-equivalent
#       to the masked full-vocab path (PR-validated TV ~0.01). Self-contained
#       Python over functions/kernels that all exist at 0.6.12
#       (topk.top_k(sorted=,deterministic=), top_p_sampling_from_probs(return_valid=)).
#       0.6.12 already carries the @flashinfer_api(trace=...) decorators, so the
#       overlay only INSERTS the fast-path blocks and leaves the decorators intact.
#   * PR #3624 (sampling.cuh): fix a shared-memory race (missing __syncthreads
#       after BlockReduce temp_storage reuse) + an out-of-range token-id write in
#       SamplingFromLogitsKernel. Correctness fix.
#
# This is an OVERLAY on dsv4-tiera2 — it inherits the entire b12x DeepSeek-V4
# stack unchanged and only replaces two flashinfer files. Output tag
# dsv4-tiera3:local leaves dsv4-tiera2:local untouched.
# =============================================================================

ARG BASE=dsv4-tiera2-src:local
FROM ${BASE}

# The dsv4-tiera2 production base installs its env under /opt/env. flashinfer is a
# regular site-packages dist. Resolve it at build time (robust to base path drift)
# and overlay the two patched files at their exact install locations.
ARG SP=/opt/env/lib/python3.12/site-packages

# Pre-flight: confirm the pinned flashinfer is the one we patched against, and the
# overlay target files exist (a stale base aborts loudly instead of mis-overlaying).
RUN python3 - <<'PYEOF'
import flashinfer, pathlib
v = getattr(flashinfer, "__version__", "?")
assert v == "0.6.12", f"flashinfer pin drift: image has {v}, overlay built for 0.6.12"
root = pathlib.Path(flashinfer.__file__).parent
for rel in ("sampling.py", "data/include/flashinfer/sampling.cuh"):
    p = root / rel
    assert p.exists(), f"overlay target missing in base: {p}"
print("flashinfer pin OK:", v, "root:", root)
PYEOF

# Overlay the two patched files into the live flashinfer package dir (resolved via
# the import above; we re-resolve here so the COPY target is the real install path).
COPY overlay/flashinfer/sampling.py /tmp/tiera3_sampling.py
COPY overlay/flashinfer/data/include/flashinfer/sampling.cuh /tmp/tiera3_sampling.cuh
RUN python3 - <<'PYEOF'
import flashinfer, pathlib, shutil
root = pathlib.Path(flashinfer.__file__).parent
shutil.copyfile("/tmp/tiera3_sampling.py", root / "sampling.py")
shutil.copyfile("/tmp/tiera3_sampling.cuh", root / "data/include/flashinfer/sampling.cuh")
print("overlaid sampling.py + data/include/flashinfer/sampling.cuh into", root)
PYEOF

# Bust the flashinfer JIT cache so the patched .cuh recompiles on first sampling
# call (any stale precompiled SamplingFromLogitsKernel must be rebuilt). The .py
# change needs no cache bust. Cache dirs vary by env; clear the known ones if present.
RUN python3 - <<'PYEOF'
import os, shutil, pathlib
candidates = [
    os.environ.get("FLASHINFER_JIT_DIR"),
    os.path.expanduser("~/.cache/flashinfer"),
    "/cache/huggingface/flashinfer",
]
for c in candidates:
    if c and pathlib.Path(c).exists():
        shutil.rmtree(c, ignore_errors=True)
        print("busted flashinfer JIT cache:", c)
print("JIT cache bust done")
PYEOF

# Verify: vllm + flashinfer import, the patched module compiles, and the #3461
# fast-path symbols are present (cheap import-level gate; full kernel JIT happens
# lazily at first sample on a GPU host).
RUN python3 - <<'PYEOF'
import py_compile, pathlib, flashinfer
from flashinfer import sampling
root = pathlib.Path(flashinfer.__file__).parent
py_compile.compile(str(root / "sampling.py"), doraise=True)
for sym in ("_top_k_first_fast_path", "_top_k_first_fast_path_applicable"):
    assert hasattr(sampling, sym), f"#3461 symbol missing after overlay: {sym}"
# #3624 marker present in the JIT header.
cuh = (root / "data/include/flashinfer/sampling.cuh").read_text()
assert "const uint32_t token_idx" in cuh and "__syncthreads();" in cuh, "#3624 .cuh overlay missing"
import vllm
print("dsv4-tiera3 overlay OK — vllm", getattr(vllm, "__version__", "?"),
      "flashinfer", flashinfer.__version__, "(#3461 sampling.py + #3624 sampling.cuh)")
PYEOF

# Entrypoint/CMD inherited from the base; the fleet launcher supplies `vllm serve`.
