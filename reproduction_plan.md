# Gymnasium Environment Re-Implementation Checklist

This plan reflects the current implementation choices while preserving the original paper's environment logic and keeping remaining work items intact.

Paper target:

- Xiong, Z., Liu, X.Y., Zhong, S., Yang, H., and Walid, A. (2018).
*Practical Deep Reinforcement Learning Approach for Stock Trading*.

---

## 1) Objective

Rebuild the original `rlstock` environment as a clean Gymnasium environment that:

- matches the paper's state/action/reward semantics,
- uses local project data paths,
- is compatible with modern RL libraries (e.g., Stable-Baselines3).

This plan intentionally avoids legacy environment injection into `site-packages/gym`.

---

## Progress Checklist

- Port environment to Gymnasium API in `src/stock_env.py`
- Consolidate data loading/preprocessing in `src/data_loader.py`
- Register train/test env IDs in `src/registration.py`
- Add per-module smoke test entrypoints (`main()`)
- Run smoke checks for train/test env behavior
- Run SB3 smoke training with DDPG
- Add validation split configuration from paper/date protocol
- Register validation env (`RLStockValidation-v0`)
- Run validation-stage hyperparameter selection
- Add min-variance benchmark for comparisons
- Final reproducibility documentation/artifacts cleanup

---

## 2) Reference Files (from original project)

Use these files as source-of-truth for business logic:

- `DQN-DDPG_Stock_Trading/DQN-DDPG_Stock_Trading/gym/envs/rlstock/rlstock_env.py`
- `DQN-DDPG_Stock_Trading/DQN-DDPG_Stock_Trading/gym/envs/rlstock/rlstock_testenv.py`
- `DQN-DDPG_Stock_Trading/DQN-DDPG_Stock_Trading/run.py`
- `README.md` (original workflow intent)

Input datasets:

- `dow_jones_30_daily_price.csv`
- `^DJI.csv`
- `dow_jones_30_ticker.txt`

---

## 3) Current Project Layout (Implemented Choices)

Implemented structure in this repo:

```text
COMP-579-Project/
  data/
    dow_jones_30_daily_price.csv
    ^DJI.csv
    dow_jones_30_ticker.txt
  src/
    stock_env.py
    data_loader.py
    registration.py
  reproduction_plan.md
```

Notes:

- Data loading + preprocessing are consolidated into `src/data_loader.py`.
- A single configurable env class (`StockEnv`) supports both train/test via constructor args.
- Registration is handled in `src/registration.py`.

---

## 4) Environment Spec to Preserve

From the original environment:

- **State**: `[cash] + [adj close prices for selected stocks] + [current holdings]`
- **Action**: per-stock buy/sell signal with bounded magnitude
- **Reward**: change in portfolio value between steps
- **Episode**: one pass over ordered trading days
- **Initialization**: fixed initial cash and zero holdings

Keep these semantics unchanged before optimization/tuning.

---

## 5) Gymnasium API Migration Requirements

Implemented with Gymnasium API:

- `reset(self, *, seed=None, options=None) -> (obs, info)`
- `step(self, action) -> (obs, reward, terminated, truncated, info)`
- uses `gymnasium.Env`
- uses explicit `np.float32` observation dtype

Current implemented action definition:

- `action_space = spaces.Box(low=-max_shares_per_trade, high=max_shares_per_trade, shape=(N_STOCKS,), dtype=np.int8)`
- `observation_space = spaces.Box(..., dtype=np.float32)`

Deviation from original draft recommendation:

- The plan originally proposed normalized float actions `[-1, 1]`; current implementation keeps integer share actions in `[-K, K]` to stay closer to original `rlstock` trade semantics.

---

## 6) Data Pipeline Checklist

### Step 6.1: Load raw data

- CSV loading is local-path based through `src/data_loader.py` (`load_prices`)

### Step 6.2: Reproduce preprocessing

- Filter tickers/date windows
- Compute adjusted prices (`adjcp`)
- Build chronological per-day slices

### Step 6.3: Split by mode

- `train` split implemented in `build_daily_frames`
- `test` split implemented in `build_daily_frames`
- `validation` split configuration
  - Confirm train/validation/trade date ranges from paper + original implementation notes
  - Add `mode="validation"` support in `build_daily_frames`
  - Add smoke check for validation slice shape/length

---

## 7) Core Implementation Checklist

### Task A: Configuration strategy

- Dynamic constructor-based configuration (no separate `stock_env_config.py`)
- Keep defaults centralized/documented for reproducibility (final README + plan pass)

### Task B: Data layer

- `load_prices(path)`
- `build_daily_frames(df, mode)`
- Remove pandas boolean-mask reindex warning in preprocessing

### Task C: `stock_env.py`

- Environment state creation
- Buy/sell execution
- Reward calculation
- `asset_memory` tracking
- Terminal handling + info dict
- DJI benchmark growth for test-mode plotting
- Add min-variance benchmark growth for test-mode comparison (same style as DJI growth, computed from episode dates)

### Task D: `registration.py`

- Register `RLStockTrain-v0`
- Register `RLStockTest-v0`
- Guard against duplicate registration
- Add `RLStockValidation-v0` after validation split ranges are finalized

---

## 8) Validation and Smoke Test Checklist

- Env runs in both `train` and `test` modes through `src/stock_env.py` `main`
- Random-step execution reaches terminal for both modes
- Runtime checks in smoke flow:
  - observation shape/dtype stability
  - finite rewards/observations
  - cash/holdings invariants
  - action input robustness
  - asset-memory/portfolio-value consistency
- Validation-mode smoke checks after split is implemented

Testing policy for this project:

- No separate test-suite files
- Per-module `main()` smoke entrypoints
- API/reward/invariant checks embedded in smoke flows

---

## 9) Integration with SB3 Checklist

- SB3 smoke run completed (`DDPG` + `MlpPolicy` + `DummyVecEnv`)
- Tiny timesteps smoke training executed successfully
- Validation-stage hyperparameter sweep (using validation split once added)
- Final train/test protocol run for report-quality results

---

## 9.1) Benchmark Comparison Checklist

- Keep DJI buy-and-hold comparison in test/trade outputs
- Add min-variance portfolio benchmark (long-only, fully invested)
- Ensure benchmark uses same stock universe and same test dates as env episode
- Rebalance at a documented cadence (e.g., monthly) using only historical data up to rebalance date
- Report side-by-side metrics: agent vs DJI vs min-variance

---

## 10) Reproducibility Checklist

- Fixed random seeds
- Data source and split ranges documented
- Environment config frozen in version control (constructor defaults + run args)
- Metrics computation script versioned
- Benchmark assumptions documented (min-variance estimator, constraints, rebalance frequency)
- Output artifacts saved under `outputs/`
- Any deviations from original paper explicitly documented

---

## 11) Credit / Attribution

This environment re-implementation should credit:

- Original project: [hust512/DQN-DDPG_Stock_Trading](https://github.com/hust512/DQN-DDPG_Stock_Trading)
- Original paper: arXiv / NeurIPS workshop publication
- Modern environment API: [Farama Gymnasium](https://gymnasium.farama.org/)

