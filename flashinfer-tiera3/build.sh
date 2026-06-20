#!/usr/bin/env bash
# Build dsv4-tiera3:local — flashinfer sampler overlay on dsv4-tiera2.
#
# Layers two merged-upstream flashinfer fixes onto the dsv4-tiera2 image:
#   * #3461 sampling.py  — top_k_first large-vocab small-k fast path (hot-path speedup)
#   * #3624 sampling.cuh — SamplingFromLogitsKernel race + OOB token-id fix
# flashinfer JIT-compiles its kernels at runtime, so (unlike vLLM csrc changes on
# the b12x prebuilt base) these overlays DO take runtime effect. See ../DSV4_TIERA3.md.
#
# Does NOT touch dsv4-tiera2:local (the running prod image) — output is a new tag.
set -euo pipefail
cd "$(dirname "$0")"

IMG="${IMG:-dsv4-tiera3:local}"
# Default base is the source-reproduced dsv4-tiera2 (../dsv4-tiera2-build). To layer
# directly on the running prod image instead, pass BASE=dsv4-tiera2:local.
BASE="${BASE:-dsv4-tiera2-src:local}"

echo "==> base present? ($BASE)"
docker image inspect "$BASE" >/dev/null 2>&1 || {
  echo "missing base $BASE — build it first: (cd ../dsv4-tiera2-build && ./build.sh)"
  echo "   or pass BASE=dsv4-tiera2:local to overlay the running prod image directly."
  exit 1
}

echo "==> building $IMG (flashinfer overlay) FROM $BASE"
DOCKER_BUILDKIT=1 docker build --network=host --build-arg "BASE=$BASE" -t "$IMG" .

echo "==> built $IMG"
docker image inspect "$IMG" --format 'id={{.Id}} size={{.Size}}'
