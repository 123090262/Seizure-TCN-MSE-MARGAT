#!/usr/bin/env bash
set -euo pipefail

# Run this script on the server login node. It waits until the next 02:00
# Asia/Shanghai time, then submits one chained LOPO job.

PROJECT_DIR="/workspace/project"
LOG_DIR="/workspace/output/scheduled_submissions"
JOB_NAME="lopo_mixedval_chb04_chb05"
SCHEDULE_TZ="Asia/Shanghai"

mkdir -p "$LOG_DIR"

now_epoch="$(TZ="$SCHEDULE_TZ" date +%s)"
target_epoch="$(TZ="$SCHEDULE_TZ" date -d "today 02:00" +%s)"
if [ "$now_epoch" -ge "$target_epoch" ]; then
  target_epoch="$(TZ="$SCHEDULE_TZ" date -d "tomorrow 02:00" +%s)"
fi

sleep_seconds="$((target_epoch - now_epoch))"
target_text="$(TZ="$SCHEDULE_TZ" date -d "@$target_epoch" "+%Y-%m-%d %H:%M:%S %Z")"
echo "[$(TZ="$SCHEDULE_TZ" date "+%Y-%m-%d %H:%M:%S %Z")] Waiting ${sleep_seconds}s until ${target_text}."
sleep "$sleep_seconds"

run_date="$(TZ="$SCHEDULE_TZ" date "+%Y-%m-%d")"
lock_dir="${LOG_DIR}/${JOB_NAME}_${run_date}.lock"
log_file="${LOG_DIR}/${JOB_NAME}_${run_date}.log"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[$(TZ="$SCHEDULE_TZ" date "+%Y-%m-%d %H:%M:%S %Z")] Skip: ${JOB_NAME} was already submitted or is being submitted for ${run_date}." | tee -a "$log_file"
  exit 0
fi

cleanup_lock() {
  status="$?"
  if [ "$status" -ne 0 ]; then
    rmdir "$lock_dir" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_lock EXIT

submit_cmd='cd /workspace/project && /usr/bin/python -m src.train configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml --fold chb04 && /usr/bin/python -m src.train configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml --fold chb05'

{
  echo "[$(TZ="$SCHEDULE_TZ" date "+%Y-%m-%d %H:%M:%S %Z")] Submitting ${JOB_NAME}."
  echo "lab-submit pytorch 8 bash -lc \"$submit_cmd\""
  cd "$PROJECT_DIR"
  lab-submit pytorch 8 bash -lc "$submit_cmd"
  echo "[$(TZ="$SCHEDULE_TZ" date "+%Y-%m-%d %H:%M:%S %Z")] lab-submit finished."
} 2>&1 | tee -a "$log_file"
