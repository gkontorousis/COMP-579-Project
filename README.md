# COMP-579 Project

Reference upstream project: [AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018)

## UV + CUDA Environment (Gymnasium + SB3)

This repo includes `pyproject.toml` and `setup_uv_cuda_venv.sh` to create a reproducible local venv with:

- Farama `gymnasium`
- `stable-baselines3`
- CUDA-enabled PyTorch wheels (Linux via `cu124` index)

Target host profile from your diagnostics:

- OS/runtime: Ubuntu 22.04
- GPU: NVIDIA H100
- Driver: 550.90.07 (CUDA 12.4 capability)

### Quick start

```bash
./setup_uv_cuda_venv.sh
```

This script will:

1. Install Python 3.11 via `uv`
2. Create `.venv`
3. Install dependencies with `uv sync --extra cuda`
4. Print a CUDA availability check from PyTorch

### Manual setup commands

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync --extra cuda
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Notes

- CUDA wheels are sourced from `https://download.pytorch.org/whl/cu124` through `pyproject.toml` (`[tool.uv.sources]`).
- If your runtime differs from CUDA 12.4, adjust the torch index in `pyproject.toml` (for example `cu121`).

## Research notes

- In the Liu paper they continue training while in the "trading" stage.
- A useful ablation is freezing policy updates during trading to measure transfer from past data vs online adaptation.
