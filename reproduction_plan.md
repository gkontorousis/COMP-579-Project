# Gymnasium Environment Re-Implementation Plan (Updated)

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
    envs/
      stock_env.py
    data_loader.py
    registration.py
  result_training.png
  result_test.png
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

## 6) Data Pipeline Plan

### Step 6.1: Load raw data

Implemented:

- CSV loading is local-path based through `src/data_loader.py` (`load_prices`).

### Step 6.2: Reproduce preprocessing

Implemented in `build_daily_frames`:

- filter tickers and date windows,
- compute adjusted prices (`adjcp`),
- construct per-day slices in chronological order.

### Step 6.3: Split by mode

Implemented:

- `mode="train"` and `mode="test"` handled in `build_daily_frames`.

Remaining optional extension (unchanged):

- `validation` split (optional but recommended).

---

## 7) Core Implementation Tasks

### Task A: Configuration strategy

Implemented choice:

- No separate `stock_env_config.py`; env is dynamically configured through constructor args (`init_balance`, `max_shares_per_trade`, paths, mode, start day).

Recommended housekeeping:

- keep defaults centralized/documented,
- treat constructor defaults as frozen experiment config for reproducibility.

### Task B: Data layer

Implemented in `src/data_loader.py`:

- `load_prices(path)`,
- `build_daily_frames(df, mode)`.

### Task C: `stock_env.py`

Implemented in `src/envs/stock_env.py`:

- environment state creation,
- buy/sell execution,
- reward calculation,
- tracking `asset_memory`,
- terminal handling and info dict,
- DJI benchmark growth calculation for test-mode plotting.

### Task D: `registration.py`

Implemented in `src/registration.py`:

- registers `RLStockTrain-v0`
- registers `RLStockTest-v0`
- checks Gymnasium registry before registering to avoid duplicate-ID errors.

---

## 8) Validation and Smoke Test Plan

Implemented smoke validation:

1. Env runs in both `train` and `test` modes through `src/envs/stock_env.py` `main`.
2. Random-step execution reaches terminal for both modes.
3. Runtime checks currently verify:
   - observation shape/dtype stability,
   - finite rewards and observations,
   - cash/holdings invariants,
   - action input robustness (sampled, float-cast, reshaped),
   - asset-memory/portfolio-value consistency.

Remaining (non-blocking cleanup):

- remove pandas boolean-mask reindex warning in `src/data_loader.py` preprocessing pipeline.

Before training any RL model (kept as target criteria):

1. Instantiate env and call `reset`.
2. Step with random actions for 10-20 steps.
3. Confirm:
   - observation shape is stable,
   - reward is finite,
   - cash/holdings never violate constraints,
   - episode terminates at expected end date.

Testing policy for this project:

- no separate test-suite files,
- each module should include a `main()` smoke entrypoint for runtime verification,
- API, reward, and invariant checks should be implemented inside module-level smoke flows.

---

## 9) Integration Plan with SB3 (Remaining)

Use this only after Section 8 passes:

- algorithm: DDPG (or TD3/SAC for robustness experiments),
- policy: `MlpPolicy`,
- vector wrapper: `DummyVecEnv`,
- action noise for exploration.

Start with tiny smoke run (`1e3-1e4` timesteps) before any long run.

---

## 10) Reproducibility Checklist (Remaining)

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

