from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RunConfig:
    # run config class to make running different seeds, algorithms (TD3 vs DDPG) and other different cli params easier
    # also needed to cleanly pass the directory to save the figures from the env during training and testing
    algo_name: str
    seed: int
    model_out: Path

    def __post_init__(self) -> None:
        self.model_out = Path(self.model_out)

    @property
    def online_model_out(self) -> Path:
        return self.model_out.parent / (self.model_out.name + "_online")

    @property
    def figure_dir(self) -> Path:
        return self.model_out.parent

    @property
    def train_figure_out(self) -> Path:
        return self.figure_dir / f"{self.model_out.name}_training.png"

    @property
    def test_figure_out(self) -> Path:
        return self.figure_dir / f"{self.model_out.name}_test.png"

    @property
    def metrics_out(self) -> Path:
        return self.figure_dir / f"{self.model_out.name}_metrics.json"

    def mkdir(self) -> None:
        """Create the output directory (idempotent)."""
        self.figure_dir.mkdir(parents=True, exist_ok=True)
