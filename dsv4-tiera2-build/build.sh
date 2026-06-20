#!/usr/bin/env bash
# Build dsv4-tiera2-src:local — the source-reproduced DeepSeek-V4 (b12x) image.
# Reproduces production dsv4-tiera2:local from the pristine Aiden base + the
# nested-layout dsv4-tiera2 patch. See Dockerfile for the layout-reconciliation
# rationale. Does NOT touch dsv4-tiera2:local (different output tag).
set -euo pipefail
cd "$(dirname "$0")"

IMG="${IMG:-dsv4-tiera2-src:local}"
BASE="aidendle94/sparkrun-vllm-ds4-gb10:production-ready"

echo "==> base present?"
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "missing base $BASE"; exit 1; }

echo "==> building $IMG (linux/arm64, b12x binary base + nested source patch)"
DOCKER_BUILDKIT=1 docker build --network=host -t "$IMG" .

echo "==> built $IMG"
docker image inspect "$IMG" --format 'id={{.Id}} size={{.Size}}'
