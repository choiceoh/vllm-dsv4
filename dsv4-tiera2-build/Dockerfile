# syntax=docker/dockerfile:1
# =============================================================================
# dsv4-tiera2-src — reproduce the Deneb production DeepSeek-V4 (b12x) image from
# the reviewable `dsv4-tiera2` source patch stack.
#
# WHY THIS IS AN OVERLAY, NOT A SOURCE BUILD
# ------------------------------------------
# The b12x MoE/MLA CUDA kernels (CUTLASS / DeepGEMM / FlashInfer artifacts for
# sm120/sm121) are *prebuilt binaries* that live ONLY inside the Aiden base
# image — they cannot be compiled from this (or any) public source tree. So the
# production image has always been built as a Python-source overlay on top of
# that binary base. This Dockerfile does the same, faithfully.
#
# LAYOUT RECONCILIATION (the core problem this harness solves)
# ------------------------------------------------------------
# The `dsv4-tiera2` git branch carries its DeepSeek-V4 source in the *flat*
# upstream layout (vllm/model_executor/models/deepseek_v4_mtp.py, no AMD variant)
# because it was cherry-picked onto DSV4@eb99b8b (2026-05-24, vLLM pre-refactor).
# The running production image was instead built on the Aiden b12x snapshot at
# vLLM commit 1967a5627 (2026-05-27) — three days later, on the *other* side of
# an upstream package refactor that moved the model into a *nested* package
# (vllm/models/deepseek_v4/{nvidia,amd}/...). eb99b8b and 1967a5627 are
# git-diverged (318 ahead / 127 behind): six of the patched files differ between
# the two snapshots for reasons unrelated to any patch here, and the model
# package is split per-platform in the image but single-file in git.
#
# Therefore the git branch's flat files are NOT a 1:1 drop-in for the image.
# Instead, the byte-faithful base is the *exact* snapshot the image used:
#   FROM aidendle94/sparkrun-vllm-ds4-gb10:production-ready   (== vLLM 1967a5627)
# and we reproduce the image's transformation chain (5 base-tier PRs + leakfix +
# 4 new-tier PRs) as a single reviewable unified patch expressed against the
# *nested* paths the image actually uses. Applying that patch to the pristine
# base yields a tree byte-identical to the production `dsv4-tiera2:local`.
#
# The patch (patches/dsv4-tiera2.patch) is the nested-layout source-of-record;
# DSV4_TIERA2.md on the branch is the per-PR human-readable provenance. They
# describe the same 18-file delta, one in nested patch form, one in prose.
# =============================================================================

FROM aidendle94/sparkrun-vllm-ds4-gb10:production-ready

ARG SP=/opt/env/lib/python3.12/site-packages

# The complete base->dsv4-tiera2 delta: 18 files, 54 hunks.
#   base-tier (5 PRs): #43733 #43445 #42971 #44253 #44420
#   leakfix (deneb#2289): block_pool + single_type_kv_cache_manager
#   new-tier (4 PRs):  #43961 #44821 #43991 #44603
# (#43988 intentionally skipped — see DSV4_TIERA2.md.)
COPY patches/dsv4-tiera2.patch /tmp/dsv4-tiera2.patch

# Apply with git (the base ships /opt/env/bin/git; `patch` is absent). --check
# first so a stale base aborts the build loudly instead of half-applying.
RUN cd "$SP" \
 && git apply --check -p1 /tmp/dsv4-tiera2.patch \
 && git apply       -p1 /tmp/dsv4-tiera2.patch \
 && echo "dsv4-tiera2 patch applied (18 files)"

# AST gate: every patched file must still parse (catches a corrupt hunk before
# it ever reaches a serve). Mirrors the gate in the legacy ~/tiera2_build chain.
RUN python3 - <<'PYEOF'
import ast, subprocess
sp = "/opt/env/lib/python3.12/site-packages"
# Files touched by the patch, recomputed from the patch headers for fidelity.
files = []
for line in open("/tmp/dsv4-tiera2.patch"):
    if line.startswith("+++ b/"):
        files.append(line[6:].strip())
assert len(files) == 18, f"expected 18 patched files, patch lists {len(files)}"
for f in files:
    p = f"{sp}/{f}"
    ast.parse(open(p).read(), filename=p)
    print("AST OK", f)
PYEOF

# Import smoke: vllm must import cleanly through every patched module path.
RUN python3 -c "import vllm; print('vllm import OK', getattr(vllm, '__version__', '?'))"

# Entrypoint/CMD are inherited from the base (null / ["sleep","infinity"]); the
# fleet launcher supplies the real `vllm serve ...` command. Do not override.
