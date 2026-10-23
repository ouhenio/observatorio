#!/usr/bin/env bash
# Wait for a run's GPU stages (process group in logs/gpu_<run>.pgid) to exit,
# then start its CPU stages if the GPU run finished cleanly.
#   setsid nohup scripts/after_gpu.sh wc100k > $OBS_ROOT/logs/cpu_wc100k.log 2>&1 &
set -uo pipefail
RUN=${1:?usage: after_gpu.sh <run_id>}
cd "$(dirname "$0")/.." && source env.sh
LOGS=$OBS_ROOT/logs
PGID=$(cat "$LOGS/gpu_$RUN.pgid")
echo "[$(date -u +%FT%TZ)] waiting for GPU process group $PGID ($RUN)"
while kill -0 -- "-$PGID" 2>/dev/null; do sleep 60; done
if grep -q "all GPU stages done" "$LOGS/gpu_$RUN.log"; then
  exec "$(dirname "$0")/run_cpu.sh" "$RUN"
fi
echo "[$(date -u +%FT%TZ)] GPU run did not finish cleanly; CPU stages not started"
exit 1
