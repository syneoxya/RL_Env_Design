from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pm_env.get_data_dir import get_env_data_dir
from pm_env.scoring_script import (
    FEE_RATE,
    LIQUIDATION_PENALTY_RATE,
    MAX_INVENTORY,
    aggregate_metrics,
    expert_action_provider,
    fill_probabilities,
    make_state,
    max_drawdown,
    model_action_provider,
    rollout_dataset,
    safe_sharpe,
    score_from_metrics,
)


STATE_DIM = 10
ACTION_DIM = 2
MIN_OFFSET = 0.002
MAX_OFFSET = 0.250


POLICY_TEMPLATE = '''import torch
from torch import nn


MIN_OFFSET = 0.002
MAX_OFFSET = 0.250


class FlowHFTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(10))
        self.register_buffer("state_std", torch.ones(10))
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -2.3))

    def raw_mean(self, x):
        x = (x - self.state_mean) / self.state_std.clamp_min(1e-6)
        return self.net(x)

    def forward(self, x):
        mean = self.raw_mean(x)
        return MIN_OFFSET + (MAX_OFFSET - MIN_OFFSET) * torch.sigmoid(mean)
'''


class FlowHFTPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_std", torch.ones(STATE_DIM))
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -2.3))

    def raw_mean(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.state_mean) / self.state_std.clamp_min(1e-6)
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.raw_mean(x)
        return MIN_OFFSET + (MAX_OFFSET - MIN_OFFSET) * torch.sigmoid(mean)

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean_action = self.forward(x)
        raw_mean = torch.logit(
            ((mean_action - MIN_OFFSET) / (MAX_OFFSET - MIN_OFFSET)).clamp(
                1e-5, 1.0 - 1e-5
            )
        )
        return torch.distributions.Normal(raw_mean, self.log_std.exp())

    def sample_action(
        self, state: np.ndarray, generator: torch.Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        x = torch.from_numpy(state).float().unsqueeze(0)
        dist = self.distribution(x)
        raw_action = dist.sample((1,)).squeeze(0)
        action = MIN_OFFSET + (MAX_OFFSET - MIN_OFFSET) * torch.sigmoid(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return (
            action.squeeze(0).detach().numpy().astype(np.float32),
            raw_action.squeeze(0).detach().numpy().astype(np.float32),
            float(log_prob.item()),
        )


@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray
    raw_actions: np.ndarray
    old_log_prob: float
    reward: float
    metrics: dict[str, float]


def load_visible_data(env_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    paths_file = env_dir / "public_val_paths.npz"
    if not paths_file.exists():
        raise FileNotFoundError(
            f"Missing {paths_file}. Run `uv run setup_data.py` before GRPO training."
        )

    train_states = np.load(env_dir / "train_states.npy").astype(np.float32)
    train_actions = np.load(env_dir / "train_actions.npy").astype(np.float32)
    paths = np.load(paths_file)
    public_paths = {
        "prices": paths["prices"].astype(np.float32),
        "buy_arrivals": paths["buy_arrivals"].astype(np.float32),
        "sell_arrivals": paths["sell_arrivals"].astype(np.float32),
        "regime_ids": paths["regime_ids"].astype(np.int64),
    }
    return train_states, train_actions, public_paths


def load_state_stats(env_dir: Path, train_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stats_path = env_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        return stats["state_mean"].astype(np.float32), stats["state_std"].astype(np.float32)

    return train_states.mean(axis=0).astype(np.float32), np.maximum(
        train_states.std(axis=0), 1e-6
    ).astype(np.float32)


def imitation_warm_start(
    model: FlowHFTPolicy,
    train_states: np.ndarray,
    train_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    if epochs <= 0:
        return

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_all = torch.from_numpy(train_states).float()
    y_all = torch.from_numpy(train_actions).float().clamp(MIN_OFFSET, MAX_OFFSET)
    n = x_all.shape[0]

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_all[idx])
            loss = F.mse_loss(pred, y_all[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def rollout_sampled_episode(
    model: FlowHFTPolicy,
    prices: np.ndarray,
    buy_arrivals: np.ndarray,
    sell_arrivals: np.ndarray,
    generator: torch.Generator,
) -> Trajectory:
    horizon = len(buy_arrivals)
    cash = 0.0
    inventory = 0.0
    equity_curve = [0.0]
    inventories = []
    actions = []
    states = []
    raw_actions = []
    old_log_prob = 0.0

    for t in range(horizon):
        price = float(prices[t])
        next_price = float(prices[t + 1])
        next_return = (next_price - price) / max(price, 1e-6)

        state = make_state(prices, buy_arrivals, sell_arrivals, t, inventory)
        action, raw_action, log_prob = model.sample_action(state, generator)
        bid_offset = float(np.clip(action[0], MIN_OFFSET, MAX_OFFSET))
        ask_offset = float(np.clip(action[1], MIN_OFFSET, MAX_OFFSET))

        bid_fill, ask_fill = fill_probabilities(
            bid_offset,
            ask_offset,
            float(buy_arrivals[t]),
            float(sell_arrivals[t]),
            float(next_return),
        )

        bid_price = price * (1.0 - bid_offset)
        ask_price = price * (1.0 + ask_offset)

        cash -= bid_fill * bid_price
        inventory += bid_fill
        cash += ask_fill * ask_price
        inventory -= ask_fill
        cash -= FEE_RATE * (bid_fill * bid_price + ask_fill * ask_price)
        inventory = float(np.clip(inventory, -3.0 * MAX_INVENTORY, 3.0 * MAX_INVENTORY))

        equity_curve.append(cash + inventory * next_price)
        inventories.append(inventory)
        actions.append([bid_offset, ask_offset])
        states.append(state)
        raw_actions.append(raw_action)
        old_log_prob += log_prob

    final_price = float(prices[-1])
    equity_curve[-1] = float(
        equity_curve[-1] - LIQUIDATION_PENALTY_RATE * abs(inventory) * final_price
    )

    equity_curve_np = np.asarray(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity_curve_np)
    inventories_np = np.asarray(inventories, dtype=np.float64)
    actions_np = np.asarray(actions, dtype=np.float64)
    start_price = max(float(prices[0]), 1e-6)

    metrics = {
        "final_pnl": float(equity_curve_np[-1]),
        "normalized_pnl": float(equity_curve_np[-1] / start_price),
        "sharpe": safe_sharpe(step_returns),
        "max_drawdown": max_drawdown(equity_curve_np) / start_price,
        "avg_abs_inventory": float(np.mean(np.abs(inventories_np))),
        "max_abs_inventory": float(np.max(np.abs(inventories_np))),
        "std_bid_offset": float(np.std(actions_np[:, 0])),
        "std_ask_offset": float(np.std(actions_np[:, 1])),
        "positive_pnl": float(equity_curve_np[-1] > 0.0),
    }
    action_std = 0.5 * (metrics["std_bid_offset"] + metrics["std_ask_offset"])
    reward = (
        metrics["normalized_pnl"]
        + 0.04 * metrics["sharpe"]
        - 1.25 * metrics["max_drawdown"]
        - 0.015 * metrics["avg_abs_inventory"]
        - 0.006 * metrics["max_abs_inventory"]
        + 0.20 * min(action_std / 0.020, 1.0)
        + 0.10 * metrics["positive_pnl"]
    )

    return Trajectory(
        states=np.asarray(states, dtype=np.float32),
        raw_actions=np.asarray(raw_actions, dtype=np.float32),
        old_log_prob=old_log_prob,
        reward=float(reward),
        metrics=metrics,
    )


def grpo_update(
    model: FlowHFTPolicy,
    optimizer: torch.optim.Optimizer,
    groups: list[list[Trajectory]],
    *,
    clip_eps: float,
    entropy_coef: float,
) -> dict[str, float]:
    losses = []
    rewards = []
    advantages_all = []

    for group in groups:
        group_rewards = torch.tensor([t.reward for t in group], dtype=torch.float32)
        advantages = (group_rewards - group_rewards.mean()) / group_rewards.std(
            unbiased=False
        ).clamp_min(1e-6)

        for trajectory, advantage in zip(group, advantages, strict=True):
            states = torch.from_numpy(trajectory.states).float()
            raw_actions = torch.from_numpy(trajectory.raw_actions).float()
            dist = model.distribution(states)
            log_prob = dist.log_prob(raw_actions).sum(dim=-1).sum()
            log_ratio = (log_prob - torch.tensor(trajectory.old_log_prob)).clamp(
                -20.0, 20.0
            )
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            entropy_bonus = dist.entropy().sum(dim=-1).mean()
            losses.append(policy_loss - entropy_coef * entropy_bonus)
            rewards.append(trajectory.reward)
            advantages_all.append(float(advantage.item()))

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "advantage_std": float(np.std(advantages_all)),
    }


def evaluate_policy(model: FlowHFTPolicy, paths: dict[str, np.ndarray]) -> tuple[float, dict]:
    _, policy_aggregate = rollout_dataset(
        action_provider=model_action_provider(model),
        prices=paths["prices"],
        buys=paths["buy_arrivals"],
        sells=paths["sell_arrivals"],
        regime_ids=paths["regime_ids"],
    )
    expert_aggregates = {}
    for expert_name in ["AS", "GLFT", "GLFT-drift"]:
        _, expert_aggregates[expert_name] = rollout_dataset(
            action_provider=expert_action_provider(expert_name),
            prices=paths["prices"],
            buys=paths["buy_arrivals"],
            sells=paths["sell_arrivals"],
            regime_ids=paths["regime_ids"],
        )
    return score_from_metrics(policy_aggregate, expert_aggregates)


def save_artifacts(model: FlowHFTPolicy, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy.py").write_text(POLICY_TEMPLATE, encoding="utf-8")
    torch.save(model.state_dict(), output_dir / "flowhft_policy.pt")


def train_grpo(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_dir = Path(args.env_dir or get_env_data_dir())
    output_dir = Path(args.output_dir or env_dir)
    train_states, train_actions, paths = load_visible_data(env_dir)
    state_mean, state_std = load_state_stats(env_dir, train_states)

    model = FlowHFTPolicy()
    model.state_mean.copy_(torch.from_numpy(state_mean))
    model.state_std.copy_(torch.from_numpy(state_std))

    imitation_warm_start(
        model,
        train_states,
        train_actions,
        epochs=args.imitation_epochs,
        batch_size=args.batch_size,
        lr=args.imitation_lr,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.grpo_lr, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed + 17)
    num_episodes = int(paths["prices"].shape[0])
    history = []

    for step in range(args.grpo_steps):
        groups = []
        selected = torch.randint(
            low=0,
            high=num_episodes,
            size=(args.episodes_per_step,),
            generator=generator,
        ).tolist()

        with torch.no_grad():
            for episode_idx in selected:
                group = [
                    rollout_sampled_episode(
                        model,
                        paths["prices"][episode_idx],
                        paths["buy_arrivals"][episode_idx],
                        paths["sell_arrivals"][episode_idx],
                        generator,
                    )
                    for _ in range(args.group_size)
                ]
                groups.append(group)

        update_stats = grpo_update(
            model,
            optimizer,
            groups,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
        )

        if step % args.eval_every == 0 or step == args.grpo_steps - 1:
            score, components = evaluate_policy(model, paths)
            row = {
                "step": step,
                "visible_score": score,
                **update_stats,
                "pnl_score": components["pnl_score"],
                "sharpe_score": components["sharpe_score"],
                "drawdown_score": components["drawdown_score"],
                "inventory_score": components["inventory_score"],
                "robustness_score": components["robustness_score"],
                "adaptivity_score": components["adaptivity_score"],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))

    save_artifacts(model, output_dir)

    final_score, final_components = evaluate_policy(model, paths)
    summary = {
        "output_dir": output_dir.as_posix(),
        "policy_path": (output_dir / "policy.py").as_posix(),
        "checkpoint_path": (output_dir / "flowhft_policy.pt").as_posix(),
        "visible_score": final_score,
        "component_scores": {
            key: final_components[key]
            for key in [
                "pnl_score",
                "sharpe_score",
                "drawdown_score",
                "inventory_score",
                "robustness_score",
                "adaptivity_score",
                "action_std",
            ]
        },
        "history": history,
    }
    (output_dir / "grpo_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a CPU FlowHFT policy with group-relative policy optimization."
    )
    parser.add_argument("--env-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--imitation-epochs", type=int, default=8)
    parser.add_argument("--imitation-lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grpo-steps", type=int, default=40)
    parser.add_argument("--grpo-lr", type=float, default=3e-4)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--episodes-per-step", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--eval-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not Path(args.env_dir or get_env_data_dir()).exists():
        raise FileNotFoundError("Data directory does not exist. Run `uv run setup_data.py`.")

    summary = train_grpo(args)
    print("Saved GRPO policy artifacts:")
    print(f"  policy.py: {summary['policy_path']}")
    print(f"  checkpoint: {summary['checkpoint_path']}")
    print(f"  visible_score: {summary['visible_score']:.6f}")


if __name__ == "__main__":
    main()
