#!/usr/bin/env bash
# One run end to end: extract + embed (run_gpu.sh, resumable per 1k-conversation chunk),
# then cluster → label → hierarchy → audit → publish (run_pipeline.sh FROM=cluster).
# Both children record the same process group (this script's), so one kill stops all:
#   kill -- -$(cat $OBS_ROOT/logs/gpu_<run>.pgid)
#   setsid nohup scripts/run_full.sh wc100k > $OBS_ROOT/logs/full_wc100k.log 2>&1 &
set -euo pipefail
RUN=${1:?usage: run_full.sh <run_id>}
cd "$(dirname "$0")/.."
scripts/run_gpu.sh "$RUN"
FROM=cluster scripts/run_pipeline.sh "$RUN"
echo "[$(date -u +%FT%TZ)] full run done ($RUN)"
