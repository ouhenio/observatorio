#!/usr/bin/env bash
# Every stage after extraction, for a run whose facets.parquet exists:
#   embed (GPU) → cluster (CPU) → label → hierarchy → audit (GPU) → publish (CPU).
# FROM=<stage> resumes at that stage (earlier stages' outputs must exist), e.g.
#   FROM=label setsid nohup scripts/run_pipeline.sh wc10k > ... &
# Records its own process group in logs/pipeline_<run>.pgid; stop with
#   kill -- -$(cat $OBS_ROOT/logs/pipeline_<run>.pgid)
#   setsid nohup scripts/run_pipeline.sh wc10k > $OBS_ROOT/logs/pipeline_wc10k.log 2>&1 &
set -euo pipefail
RUN=${1:?usage: run_pipeline.sh <run_id> [hierarchy args...]}
shift || true
cd "$(dirname "$0")/.."
source env.sh
ps -o pgid= -p $$ | tr -d ' ' > "$OBS_ROOT/logs/pipeline_$RUN.pgid"
STAGES=(embed cluster label hierarchy audit publish)
FROM=${FROM:-embed}
[[ " ${STAGES[*]} " == *" $FROM "* ]] || { echo "unknown FROM=$FROM (one of: ${STAGES[*]})"; exit 2; }
started=0
step() {
  [[ $1 == "$FROM" ]] && started=1
  if (( ! started )); then echo "[$(date -u +%FT%TZ)] $1 skipped (FROM=$FROM)"; return; fi
  echo "[$(date -u +%FT%TZ)] $1 start"; shift; uv run python -m "$@"
}
step embed     observatorio.embed --run "$RUN"
step cluster   observatorio.cluster --run "$RUN"
step label     observatorio.label --run "$RUN"
step hierarchy observatorio.hierarchy --run "$RUN" "$@"
step audit     observatorio.audit --run "$RUN"
step publish   observatorio.publish --run "$RUN"
echo "[$(date -u +%FT%TZ)] pipeline done ($RUN)"
