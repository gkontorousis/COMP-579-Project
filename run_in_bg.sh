#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs

SEEDS=(42 43 44)
SEED_WORKERS=3
N_EPISODES=50
N_TRIALS=15
# Keep BLAS/thread usage modest per top-level job since each job also spawns seed workers.
THREADS_PER_JOB=2
DDPG_LOG="outputs/logs/ddpg_report_${N_EPISODES}ep_${N_TRIALS}trials.log"
TD3_LOG="outputs/logs/td3_report_${N_EPISODES}ep_${N_TRIALS}trials.log"
DDPG_RUN_DIR="outputs/ddpg_report_${N_EPISODES}ep_${N_TRIALS}trials"
TD3_RUN_DIR="outputs/td3_report_${N_EPISODES}ep_${N_TRIALS}trials"

OMP_NUM_THREADS=$THREADS_PER_JOB VECLIB_MAXIMUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB \
uv run python -m src.main \
  --dj_30_dp_path data/dow_jones_30_daily_price.csv \
  --dji_path 'data/^DJI.csv' \
  --algo ddpg \
  --seeds "${SEEDS[@]}" \
  --seed_workers "$SEED_WORKERS" \
  --n_episodes "$N_EPISODES" \
  --n_trials "$N_TRIALS" \
  --model_out "${DDPG_RUN_DIR}/ddpg_model" \
  --optuna_n_jobs 3 \
  > "$DDPG_LOG" 2>&1 &

DDPG_PID=$!

OMP_NUM_THREADS=$THREADS_PER_JOB VECLIB_MAXIMUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB \
uv run python -m src.main \
  --dj_30_dp_path data/dow_jones_30_daily_price.csv \
  --dji_path 'data/^DJI.csv' \
  --algo td3 \
  --seeds "${SEEDS[@]}" \
  --seed_workers "$SEED_WORKERS" \
  --n_episodes "$N_EPISODES" \
  --n_trials "$N_TRIALS" \
  --model_out "${TD3_RUN_DIR}/td3_model" \
  --optuna_n_jobs 3 \
  > "$TD3_LOG" 2>&1 &

TD3_PID=$!

echo "DDPG PID: $DDPG_PID"
echo "TD3  PID: $TD3_PID"
echo
echo "Logs:"
echo "  DDPG: $DDPG_LOG"
echo "  TD3 : $TD3_LOG"
echo
echo "Monitor with:"
echo "  tail -f $DDPG_LOG"
echo "  tail -f $TD3_LOG"
echo
echo "Check status:"
echo "  ps -p $DDPG_PID"
echo "  ps -p $TD3_PID"
echo
echo "Config:"
echo "  seeds=${SEEDS[*]} seed_workers=$SEED_WORKERS threads_per_job=$THREADS_PER_JOB"
wait $DDPG_PID $TD3_PID