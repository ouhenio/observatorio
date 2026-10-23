#!/usr/bin/env bash
# GPU stages for one run: extract (Qwen3.8-27B) → embed (Qwen3-Embedding-8B).
# Checkpoints every 1,000 conversations (parts/extract_NNNN.parquet); rerunning
# skips finished chunks. Keep --chunk fixed for a run, since part numbers depend on it.
# Records its own process group in logs/gpu_<run>.pgid; stop everything (incl. vLLM's engine) with
#   kill -- -$(cat $OBS_ROOT/logs/gpu_<run>.pgid)
#   setsid nohup scripts/run_gpu.sh wc100k > $OBS_ROOT/logs/gpu_wc100k.log 2>&1 &
set -euo pipefail
RUN=${1:?usage: run_gpu.sh <run_id>}
cd "$(dirname "$0")/.."
source env.sh
ps -o pgid= -p $$ | tr -d ' ' > "$OBS_ROOT/logs/gpu_$RUN.pgid"
echo "[$(date -u +%FT%TZ)] extract start ($RUN)"
uv run python -m observatorio.extract --run "$RUN" --chunk 1000
echo "[$(date -u +%FT%TZ)] extract done; embed start"
uv run python -m observatorio.embed --run "$RUN"
echo "[$(date -u +%FT%TZ)] all GPU stages done"
