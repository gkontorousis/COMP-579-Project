#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs

SEEDS=(42 43 44)
SEED_WORKERS=3
N_EPISODES=500
# Keep BLAS/thread usage modest per top-level job since each job also spawns seed workers.
THREADS_PER_JOB=2

COVID_TRAIN_DP_PATH="${COVID_TRAIN_DP_PATH:-data/covid/train/20100101_20191231_dow_jones_30_daily_price.csv}"
COVID_TRADE_DP_PATH="${COVID_TRADE_DP_PATH:-data/covid/test/20200101_20201231_dow_jones_30_daily_price.csv}"
COVID_DJI_PATH="${COVID_DJI_PATH:-data/covid/test/20200101_20201231_^DJI.csv}"

DDPG_LOG="outputs/logs/covid_ddpg_report_${N_EPISODES}ep.log"
TD3_LOG="outputs/logs/covid_td3_report_${N_EPISODES}ep.log"
DDPG_RUN_DIR="outputs/covid_ddpg_report_${N_EPISODES}ep"
TD3_RUN_DIR="outputs/covid_td3_report_${N_EPISODES}ep"

COVID_CLI=(
  --covid_train_dp_path "$COVID_TRAIN_DP_PATH"
  --covid_trade_dp_path "$COVID_TRADE_DP_PATH"
  --covid_dji_path "$COVID_DJI_PATH"
)

OMP_NUM_THREADS=$THREADS_PER_JOB VECLIB_MAXIMUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB \
uv run python -m src.covid_run \
  "${COVID_CLI[@]}" \
  --algo ddpg \
  --seeds "${SEEDS[@]}" \
  --seed_workers "$SEED_WORKERS" \
  --n_episodes "$N_EPISODES" \
  --model_out "${DDPG_RUN_DIR}/ddpg_covid_model" \
  > "$DDPG_LOG" 2>&1 &

DDPG_PID=$!

OMP_NUM_THREADS=$THREADS_PER_JOB VECLIB_MAXIMUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB \
uv run python -m src.covid_run \
  "${COVID_CLI[@]}" \
  --algo td3 \
  --seeds "${SEEDS[@]}" \
  --seed_workers "$SEED_WORKERS" \
  --n_episodes "$N_EPISODES" \
  --model_out "${TD3_RUN_DIR}/td3_covid_model" \
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
echo "  seeds=${SEEDS[*]} seed_workers=$SEED_WORKERS threads_per_job=$THREADS_PER_JOB n_episodes=$N_EPISODES"
wait $DDPG_PID $TD3_PID
