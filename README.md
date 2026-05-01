# COMP-579 Project

Reference upstream project: [AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018)

This is the repository for our COMP-579 final course project.

We chose to reproduce a [paper](pdfs/reference_papers/Liu - Practical Deep Reinforcement Learning Approach for Stock Trading.pdf) which applies reinforcement learning to stock trading and compares its performance to min-variance portfolio allocation strategy and the Dow Jones Industrial Average (DJIA).

### Repository layout
The repository is organized as follows:

```
.
├── README.md
├── pyproject.toml
├── uv.lock
├── run_in_bg.sh
├── run_in_bg_covid.sh
├── data/
│   └── covid/
├── original_files/
├── outputs/
├── pdfs/
├── selected_outputs/
└── src/
```

- **`run_in_bg.sh`** / **`run_in_bg_covid.sh`** — Helper scripts to launch the main and covid_run modules for training and obtaining results
- **`data/`** — The directory containing the csv files used for constructing the environment. The top level files are copied from the original paper's open source repository to facilitate reproduction of their results.
- **`data/covid/`** — The covid subdirectory was populated for a custom range employing the yfinance API to investigate performance over market crash
- **`original_files/`** — From the reference upstream project we copied and placed here the relevant files that we used as reference to reproduce the paper's results.
- **`pdfs/`** — Our proposal, the project description handout, and any reference papers we used (including the original paper)
- **`selected_outputs/`** — Selected results from some runs (stored as git LFS objects)
- **`src/`** — The main codebase


### Set Up

You need to install `uv` a python package manager
Once installed you simply do `uv sync` which creates the `.venv` which has all the required dependencies for running the project's code

## Running experiments
To run the experiments we have two helper scripts that launch in the background parallel runs for DDPG and TD3 RL algorithms, one for reproducing the original paper results and one for our COVID range experiment

The files `src/main.py` and `src/covid_run.py` are the main entrypoints to run each experiment. See files for available input cli args