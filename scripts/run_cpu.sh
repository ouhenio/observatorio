#!/usr/bin/env bash
# CPU stages for one run: cluster (k-means + UMAP + privacy threshold) → publish.
# Publishes with placeholder names until the label/hierarchy/audit stages exist.
#   nohup scripts/run_cpu.sh wc100k > $OBS_ROOT/logs/cpu_wc100k.log 2>&1 &
set -euo pipefail
RUN=${1:?usage: run_cpu.sh <run_id>}
cd "$(dirname "$0")/.."
source env.sh
echo "[$(date -u +%FT%TZ)] cluster start ($RUN)"
uv run python -m observatorio.cluster --run "$RUN"
echo "[$(date -u +%FT%TZ)] publish start"
uv run python -m observatorio.publish --run "$RUN"
echo "[$(date -u +%FT%TZ)] all CPU stages done"
