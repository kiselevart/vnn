#!/bin/bash
# GPU Training Launcher
# Usage: edit the JOBS array below, then run: bash launch_training.sh

# ── Define your jobs here ─────────────────────────────────────────────────────
# Format: "GPU_IDS | command"
# GPU_IDS can be a single GPU (e.g. 0) or multiple (e.g. 0,1)

JOBS=(
  "3 | python benchmark.py --model vnn_1d --suite quick"
  "3 | python benchmark.py --model vnn_1d --suite quick --disable_cubic"
  "3 | python benchmark.py --model vnn_1d --suite quick --cubic_mode general"
  "3 | python benchmark.py --model vnn_1d --suite quick --Q 4"
  "3 | python benchmark.py --model vnn_1d --suite quick --base_ch 12"

  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 1"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 2"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 3"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 1 2"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 2 3"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 1 2 3"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 2 3 4"
  "4 | python benchmark.py --model laguerre_vnn_1d --suite quick --poly_degrees 1 2 3 4"
)

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ── Launch ────────────────────────────────────────────────────────────────────
PIDS=()

for i in "${!JOBS[@]}"; do
  IFS='|' read -r gpus cmd <<< "${JOBS[$i]}"
  gpus=$(echo "$gpus" | xargs)   # trim whitespace
  cmd=$(echo "$cmd" | xargs)

  LOG_FILE="$LOG_DIR/job_${i}_gpu${gpus}.log"
  echo "Launching job $i on GPU(s) $gpus → $LOG_FILE"

  CUDA_VISIBLE_DEVICES="$gpus" bash -c "$cmd; echo; echo \"--- Command: CUDA_VISIBLE_DEVICES=$gpus $cmd ---\"" > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All ${#JOBS[@]} job(s) launched. PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_DIR/*.log        # stream all logs"
echo "  watch -n2 nvidia-smi          # watch GPU usage"
echo ""

# Wait for all jobs and report exit codes
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    echo "✓ Job $i (PID ${PIDS[$i]}) finished successfully"
  else
    echo "✗ Job $i (PID ${PIDS[$i]}) exited with code $STATUS"
  fi
done

echo ""
echo "All jobs done."