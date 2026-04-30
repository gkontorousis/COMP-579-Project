# Gymnasium Environment Re-Implementation Checklist

This plan reflects the current implementation choices while preserving the original paper's environment logic and keeping remaining work items intact.

Paper target:

- Xiong, Z., Liu, X.Y., Zhong, S., Yang, H., and Walid, A. (2018).
  _Practical Deep Reinforcement Learning Approach for Stock Trading_.

---

## 1) Objective

Rebuild the original `rlstock` environment as a clean Gymnasium environment that:

- matches the paper's state/action/reward semantics,
- uses local project data paths,
- is compatible with modern RL libraries (e.g., Stable-Baselines3).

This plan intentionally avoids legacy environment injection into `site-packages/gym`.

---

## Progress Checklist

- [X] Port environment to Gymnasium API in `src/stock_env.py`
- [X] Consolidate data loading/preprocessing in `src/data_loader.py`
- [X] Register train/test env IDs in `src/registration.py`
- [X] Add per-module smoke test entrypoints (`main()`)
- [X] Run smoke checks for train/test env behavior
- [X] Run SB3 smoke training with DDPG
- [ ] Add validation split configuration from paper/date protocol
- [ ] Register validation env (`RLStockValidation-v0`)
- [ ] Run validation-stage hyperparameter selection
- [ ] Final reproducibility documentation/artifacts cleanup

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

- [X] CSV loading is local-path based through `src/data_loader.py` (`load_prices`)

### Step 6.2: Reproduce preprocessing

- [X] Filter tickers/date windows
- [X] Compute adjusted prices (`adjcp`)
- [X] Build chronological per-day slices

### Step 6.3: Split by mode

- [X] `train` split implemented in `build_daily_frames`
- [X] `test` split implemented in `build_daily_frames`
- [ ] `validation` split configuration
  - [ ] Confirm train/validation/trade date ranges from paper + original implementation notes
  - [ ] Add `mode="validation"` support in `build_daily_frames`
  - [ ] Add smoke check for validation slice shape/length

---

## 7) Core Implementation Checklist

### Task A: Configuration strategy

- [X] Dynamic constructor-based configuration (no separate `stock_env_config.py`)
- [ ] Keep defaults centralized/documented for reproducibility (final README + plan pass)

### Task B: Data layer

- [X] `load_prices(path)`
- [X] `build_daily_frames(df, mode)`
- [ ] Remove pandas boolean-mask reindex warning in preprocessing

### Task C: `stock_env.py`

- [X] Environment state creation
- [X] Buy/sell execution
- [X] Reward calculation
- [X] `asset_memory` tracking
- [X] Terminal handling + info dict
- [X] DJI benchmark growth for test-mode plotting

### Task D: `registration.py`

- [X] Register `RLStockTrain-v0`
- [X] Register `RLStockTest-v0`
- [X] Guard against duplicate registration
- [ ] Add `RLStockValidation-v0` after validation split ranges are finalized

---

## 8) Validation and Smoke Test Checklist

- [X] Env runs in both `train` and `test` modes through `src/stock_env.py` `main`
- [X] Random-step execution reaches terminal for both modes
- [X] Runtime checks in smoke flow:
  - observation shape/dtype stability
  - finite rewards/observations
  - cash/holdings invariants
  - action input robustness
  - asset-memory/portfolio-value consistency
- [ ] Validation-mode smoke checks after split is implemented

Testing policy for this project:

- [X] No separate test-suite files
- [X] Per-module `main()` smoke entrypoints
- [X] API/reward/invariant checks embedded in smoke flows

---

## 9) Integration with SB3 Checklist

- [X] SB3 smoke run completed (`DDPG` + `MlpPolicy` + `DummyVecEnv`)
- [X] Tiny timesteps smoke training executed successfully
- [ ] Validation-stage hyperparameter sweep (using validation split once added)
- [ ] Final train/test protocol run for report-quality results

---

## 10) Reproducibility Checklist

- [ ] Fixed random seeds
- [ ] Data source and split ranges documented
- [ ] Environment config frozen in version control (constructor defaults + run args)
- [ ] Metrics computation script versioned
- [ ] Output artifacts saved under `outputs/`
- [ ] Any deviations from original paper explicitly documented

---

## 11) Credit / Attribution

This environment re-implementation should credit:

- Original project: [hust512/DQN-DDPG_Stock_Trading](https://github.com/hust512/DQN-DDPG_Stock_Trading)
- Original paper: arXiv / NeurIPS workshop publication
- Modern environment API: [Farama Gymnasium](https://gymnasium.farama.org/)
