# Gymnasium Environment Re-Implementation Plan

This plan focuses on recreating a **custom stock-trading environment** using **Farama Gymnasium**, while preserving the original paper's environment logic.

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

## 3) Target Project Layout

Recommended structure:

```text
my_project/
  data/
    dow_jones_30_daily_price.csv
    ^DJI.csv
    dow_jones_30_ticker.txt
  src/
    envs/
      stock_env.py
      stock_env_config.py
    data/
      loaders.py
      preprocess.py
    train/
      train_smoke.py
    eval/
      evaluate.py
    registration.py
  outputs/
  tests/
    test_env_api.py
    test_reward_logic.py
  README.md
```

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

Implement environment with Gymnasium API:

- `reset(self, *, seed=None, options=None) -> (obs, info)`
- `step(self, action) -> (obs, reward, terminated, truncated, info)`
- use `gymnasium.Env`
- use explicit `np.float32` observation/action dtypes

Recommended definitions:

- `action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_STOCKS,), dtype=np.float32)`
- `observation_space = spaces.Box(..., dtype=np.float32)`

Then map normalized action `[-1, 1]` to trade units internally.

---

## 6) Data Pipeline Plan

### Step 6.1: Load raw data

- Read CSVs from `my_project/data/` only (no absolute paths).

### Step 6.2: Reproduce preprocessing

Match original logic:

- filter tickers and date windows,
- compute adjusted prices (`adjcp`),
- construct per-day slices in chronological order.

### Step 6.3: Split by mode

Use config-driven ranges:

- `train`
- `validation` (optional but recommended)
- `test`

---

## 7) Core Implementation Tasks

### Task A: `stock_env_config.py`

Define reusable config:

- initial cash,
- max trade size mapping,
- train/test date boundaries,
- transaction cost toggle (if added, keep off by default for paper fidelity).

### Task B: `loaders.py` and `preprocess.py`

Create deterministic preprocessing functions:

- `load_prices(path)`,
- `build_daily_frames(df, mode, config)`.

### Task C: `stock_env.py`

Implement:

- environment state creation,
- buy/sell execution,
- reward calculation,
- tracking `asset_memory`,
- terminal handling and info dict.

### Task D: `registration.py`

Register env IDs such as:

- `RLStockTrain-v0`
- `RLStockTest-v0`

using Gymnasium registration utilities.

---

## 8) Validation and Smoke Test Plan

Before training any RL model:

1. Instantiate env and call `reset`.
2. Step with random actions for 10-20 steps.
3. Confirm:
   - observation shape is stable,
   - reward is finite,
   - cash/holdings never violate constraints,
   - episode terminates at expected end date.

Add unit tests:

- API compliance test (`reset`/`step` return signatures),
- deterministic reward sanity test on fixed synthetic mini-data.

---

## 9) Integration Plan with SB3 (after env is stable)

Use this only after Section 8 passes:

- algorithm: DDPG (or TD3/SAC for robustness experiments),
- policy: `MlpPolicy`,
- vector wrapper: `DummyVecEnv`,
- action noise for exploration.

Start with tiny smoke run (`1e3-1e4` timesteps) before any long run.

---

## 10) Reproducibility Checklist

- [ ] Fixed random seeds
- [ ] Data source and split ranges documented
- [ ] Environment config frozen in version control
- [ ] Metrics computation script versioned
- [ ] Output artifacts saved under `outputs/`
- [ ] Any deviations from original paper explicitly documented

---

## 11) Credit / Attribution

This environment re-implementation should credit:

- Original project: [hust512/DQN-DDPG_Stock_Trading](https://github.com/hust512/DQN-DDPG_Stock_Trading)
- Original paper: arXiv / NeurIPS workshop publication
- Modern environment API: [Farama Gymnasium](https://gymnasium.farama.org/)

