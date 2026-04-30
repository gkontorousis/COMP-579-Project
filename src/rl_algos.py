"""Central RL module: model construction, training, hyperparameter tuning, and online trade-phase learning."""

from pathlib import Path
import argparse
import optuna
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import gymnasium as gym
from src.registration import register_stock_envs

_ALGO_CLASSES = {
    "ddpg": DDPG,
    "td3": TD3,
}
SUPPORTED_ALGO_NAMES = tuple(sorted(_ALGO_CLASSES))


def get_algo_class(name: str):
    key = (name or "ddpg").lower().strip()
    cls = _ALGO_CLASSES.get(key)
    if cls is None:
        choices = ", ".join(sorted(_ALGO_CLASSES))
        raise ValueError(f"Unknown algo {name!r}; supported: {choices}")
    return cls


def _build_model(algo, env, seed, sigma, verbose, **kwargs):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))
    # [400, 300] follows Lillicrap et al. (2015); TD3 uses the same actor-critic widths in SB3
    policy_kwargs = dict(net_arch=[400, 300])
    return algo(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=verbose,
        seed=seed,
        policy_kwargs=policy_kwargs,
        **kwargs,
    )


def _set_figure_out(vec_env, figure_out) -> None:
    """Push a figure output path onto every inner StockEnv in a VecEnv."""
    if figure_out is None:
        return
    figure_out = Path(figure_out)
    for e in vec_env.envs:
        e.unwrapped.figure_out = figure_out


def train(algo, env, timesteps, seed, model_out, sigma=0.1, figure_out=None, **kwargs):
    """Build and train a model from scratch, save to model_out."""
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    _set_figure_out(env, figure_out)
    model = _build_model(algo, env, seed, sigma, verbose=1, **kwargs)
    model.learn(total_timesteps=timesteps)
    model.save(str(model_out))


def tune(algo, train_env_id, val_env_id, timesteps_per_trial, seed, n_trials):

    # use Optuna (https://optuna.org) for hyperparameter optimization
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        tau = trial.suggest_float("tau", 0.001, 0.01)
        sigma = trial.suggest_float("noise_sigma", 0.05, 0.3)

        train_env = DummyVecEnv([lambda: gym.make(train_env_id)])
        model = _build_model(
            algo,
            train_env,
            seed,
            sigma,
            verbose=0,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
        )
        model.learn(total_timesteps=timesteps_per_trial)
        train_env.close()

        val_env = DummyVecEnv([lambda: gym.make(val_env_id)])
        obs = val_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = val_env.step(action)
            total_reward += float(rewards[0])
            done = bool(dones[0])
        val_env.close()
        return total_reward

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def trade_online(algo, model_path, env, model_out, figure_out=None):
    """Load a trained model and continue training for one full test episode (online learning).

    As per the paper: 'we continue training our agent while in the trading stage as this
    will improve the agent to better adapt the market dynamics.'

    The env's step() saves the result figure on episode termination.
    Saves the online-updated model to model_out.
    """
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    _set_figure_out(env, figure_out)
    model = algo.load(str(model_path), env=env)
    timesteps = env.envs[0].unwrapped.terminal_day + 1
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(str(model_out))


def main():
    parser = argparse.ArgumentParser("RLStock rl_algos smoke test", allow_abbrev=False)
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=Path, default=Path("outputs/rl_algos_smoke"))
    parser.add_argument(
        "--algo",
        type=str,
        default="ddpg",
        choices=SUPPORTED_ALGO_NAMES,
    )
    args = parser.parse_args()
    algo_cls = get_algo_class(args.algo)

    register_stock_envs(data_path=args.dj_30_dp_path, dji_path=args.dji_path)

    train_env = DummyVecEnv([lambda: gym.make("RLStockTrain-v0")])
    train(algo_cls, train_env, args.timesteps, args.seed, args.model_out)
    train_env.close()

    test_env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
    online_out = args.model_out.parent / (args.model_out.name + "_online")
    trade_online(algo_cls, args.model_out, test_env, online_out)
    test_env.close()
    print(f"Smoke test OK. Online model saved to {online_out}")


if __name__ == "__main__":
    main()
