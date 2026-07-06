# setup_data.py
# Builds the visible and hidden FlowHFT-style data.
#
# Hidden regimes are held out for scoring.

# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy>=1.26",
# ]
# ///

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RANDOM_SEED = 42

STATE_DIM = 10
ACTION_DIM = 2
MAX_INVENTORY = 20.0
HORIZON = 200

EXPERT_POOL = ["AS", "GLFT", "GLFT-drift"]


def hurst_to_autocorr(hurst: float) -> float:
    if hurst < 0.5:
        return -0.35
    if hurst > 0.5:
        return 0.35
    return 0.0


VISIBLE_REGIMES: list[dict[str, Any]] = [
    {
        "name": "visible_LH_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LH",
        "teacher_expert": "GLFT",
    },
    {
        "name": "visible_LL_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.02,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LL",
        "teacher_expert": "GLFT",
    },
    {
        "name": "visible_HH_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HH",
        "teacher_expert": "AS",
    },
    {
        "name": "visible_HL_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.02,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HL",
        "teacher_expert": "AS",
    },
    {
        "name": "visible_LH_trending_up",
        "hurst": 0.8,
        "drift": 0.10,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LH",
        "teacher_expert": "GLFT-drift",
    },
    {
        "name": "visible_LH_trending_down",
        "hurst": 0.8,
        "drift": -0.10,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LH",
        "teacher_expert": "GLFT-drift",
    },
]


HIDDEN_REGIMES: list[dict[str, Any]] = [
    {
        "name": "hidden_LH_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LH",
        "teacher_expert": "GLFT",
    },
    {
        "name": "hidden_LL_mean_reverting",
        "hurst": 0.2,
        "drift": 0.0,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.02,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LL",
        "teacher_expert": "GLFT",
    },
    {
        "name": "hidden_HH_trending_up",
        "hurst": 0.8,
        "drift": 0.20,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HH",
        "teacher_expert": "GLFT-drift",
    },
    {
        "name": "hidden_HL_trending_down",
        "hurst": 0.8,
        "drift": -0.20,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.01,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HL",
        "teacher_expert": "GLFT-drift",
    },
    {
        "name": "hidden_HH_mean_reverting",
        "hurst": 0.2,
        "drift": 0.0,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.01,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HH",
        "teacher_expert": "AS",
    },
    {
        "name": "hidden_HL_random",
        "hurst": 0.5,
        "drift": 0.0,
        "vol": 0.25,
        "jump_mean": 0.02,
        "jump_prob": 0.020,
        "dt": 0.02,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "HL",
        "teacher_expert": "AS",
    },
    {
        "name": "hidden_LH_trending_up",
        "hurst": 0.8,
        "drift": 0.20,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.02,
        "arrival_rate": 50.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LH",
        "teacher_expert": "GLFT-drift",
    },
    {
        "name": "hidden_LL_trending_down",
        "hurst": 0.8,
        "drift": -0.20,
        "vol": 0.02,
        "jump_mean": 0.02,
        "jump_prob": 0.010,
        "dt": 0.02,
        "arrival_rate": 25.0,
        "self_exciting": 0.7,
        "cross_exciting": 0.3,
        "market_type": "LL",
        "teacher_expert": "GLFT-drift",
    },
]


def simulate_price_process(
    regime: dict[str, Any],
    rng: np.random.Generator,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    prices = np.empty(horizon + 1, dtype=np.float32)
    returns = np.empty(horizon, dtype=np.float32)

    prices[0] = float(rng.uniform(80.0, 120.0))

    hurst = float(regime.get("hurst", 0.5))
    drift = float(regime.get("drift", 0.0))
    vol = float(regime.get("vol", 0.02))
    jump_mean = float(regime.get("jump_mean", 0.02))
    jump_prob = float(regime.get("jump_prob", 0.01))
    dt = float(regime.get("dt", 0.01))

    return_autocorr = hurst_to_autocorr(hurst)
    prev_return = 0.0

    for t in range(horizon):
        shock = vol * np.sqrt(dt) * rng.normal()
        drift_component = drift * dt

        jump = 0.0
        if rng.random() < jump_prob:
            jump_direction = 1.0 if rng.random() < 0.5 else -1.0
            jump = jump_direction * jump_mean

        ret = drift_component + return_autocorr * prev_return + shock + jump

        if hurst < 0.5:
            price_deviation = prices[t] / max(prices[0], 1e-6) - 1.0
            ret -= 0.10 * price_deviation * dt

        ret = float(np.clip(ret, -0.15, 0.15))

        prices[t + 1] = max(1.0, prices[t] * (1.0 + ret))
        returns[t] = ret
        prev_return = ret

    return prices, returns


def simulate_order_arrivals(
    regime: dict[str, Any],
    returns: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = len(returns)

    arrival_rate = float(regime.get("arrival_rate", 25.0))
    self_exciting = float(regime.get("self_exciting", 0.7))
    cross_exciting = float(regime.get("cross_exciting", 0.3))

    buy_arrivals = np.empty(horizon, dtype=np.float32)
    sell_arrivals = np.empty(horizon, dtype=np.float32)

    prev_buy = arrival_rate
    prev_sell = arrival_rate

    for t in range(horizon):
        flow_signal = np.tanh(30.0 * float(returns[t]))

        buy_base = arrival_rate * np.exp(0.30 * flow_signal)
        sell_base = arrival_rate * np.exp(-0.30 * flow_signal)

        buy_intensity = (
            buy_base
            + 0.10 * self_exciting * prev_buy
            + 0.05 * cross_exciting * prev_sell
        )
        sell_intensity = (
            sell_base
            + 0.10 * self_exciting * prev_sell
            + 0.05 * cross_exciting * prev_buy
        )

        buy_intensity = float(np.clip(buy_intensity, 1.0, 3.0 * arrival_rate))
        sell_intensity = float(np.clip(sell_intensity, 1.0, 3.0 * arrival_rate))

        buy_count = rng.poisson(buy_intensity)
        sell_count = rng.poisson(sell_intensity)

        buy_arrivals[t] = float(buy_count)
        sell_arrivals[t] = float(sell_count)

        prev_buy = 0.70 * prev_buy + 0.30 * buy_count
        prev_sell = 0.70 * prev_sell + 0.30 * sell_count

    return buy_arrivals, sell_arrivals


def simulate_market_path(
    regime: dict[str, Any],
    rng: np.random.Generator,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices, returns = simulate_price_process(regime, rng, horizon=horizon)
    buy_arrivals, sell_arrivals = simulate_order_arrivals(regime, returns, rng)
    return prices, buy_arrivals, sell_arrivals


def make_state(
    prices: np.ndarray,
    buy_arrivals: np.ndarray,
    sell_arrivals: np.ndarray,
    t: int,
    inventory: float,
    max_inventory: float = MAX_INVENTORY,
) -> np.ndarray:
    price = float(prices[t])
    prev_price = float(prices[max(0, t - 1)])
    start_price = float(prices[0])

    ret_1 = (price - prev_price) / max(prev_price, 1e-6)

    start = max(0, t - 10)
    price_window = prices[start : t + 1]

    if len(price_window) > 2:
        returns = np.diff(price_window) / np.maximum(price_window[:-1], 1e-6)
        ret_mean = float(np.mean(returns))
        realized_vol = float(np.std(returns))
    else:
        ret_mean = 0.0
        realized_vol = 0.0

    if t > start:
        buy_recent = float(np.mean(buy_arrivals[start:t]))
        sell_recent = float(np.mean(sell_arrivals[start:t]))
    else:
        buy_recent = float(buy_arrivals[t])
        sell_recent = float(sell_arrivals[t])

    imbalance = (buy_recent - sell_recent) / (buy_recent + sell_recent + 1e-6)
    time_frac = t / max(1, len(buy_arrivals) - 1)
    remaining_time = 1.0 - time_frac
    price_dev = price / max(start_price, 1e-6) - 1.0

    return np.array(
        [
            inventory / max_inventory,
            time_frac,
            ret_1,
            ret_mean,
            realized_vol,
            buy_recent / 50.0,
            sell_recent / 50.0,
            imbalance,
            price_dev,
            remaining_time,
        ],
        dtype=np.float32,
    )


def as_expert_action(state: np.ndarray) -> np.ndarray:
    inventory = float(state[0])
    realized_vol = abs(float(state[4]))
    remaining_time = float(state[9])

    base_spread = 0.026 + 2.40 * realized_vol + 0.006 * remaining_time
    inventory_penalty = 0.060

    bid = base_spread + inventory_penalty * max(inventory, 0.0)
    ask = base_spread + inventory_penalty * max(-inventory, 0.0)

    return np.array([bid, ask], dtype=np.float32)


def glft_expert_action(state: np.ndarray) -> np.ndarray:
    inventory = float(state[0])
    realized_vol = abs(float(state[4]))
    imbalance = float(state[7])

    base_spread = 0.012 + 1.05 * realized_vol
    inventory_penalty = 0.030
    imbalance_skew = 0.020 * imbalance

    bid = base_spread + inventory_penalty * max(inventory, 0.0) - imbalance_skew
    ask = base_spread + inventory_penalty * max(-inventory, 0.0) + imbalance_skew

    return np.array([bid, ask], dtype=np.float32)


def glft_drift_expert_action(state: np.ndarray) -> np.ndarray:
    inventory = float(state[0])
    ret_1 = float(state[2])
    ret_mean = float(state[3])
    realized_vol = abs(float(state[4]))
    imbalance = float(state[7])

    base_spread = 0.014 + 1.15 * realized_vol
    inventory_penalty = 0.032

    drift_signal = np.clip(
        1.20 * ret_1 + 3.50 * ret_mean + 0.012 * imbalance,
        -0.035,
        0.035,
    )

    bid = base_spread + inventory_penalty * max(inventory, 0.0) - drift_signal
    ask = base_spread + inventory_penalty * max(-inventory, 0.0) + drift_signal

    return np.array([bid, ask], dtype=np.float32)


def expert_action_by_name(name: str, state: np.ndarray) -> np.ndarray:
    if name == "AS":
        action = as_expert_action(state)
    elif name == "GLFT":
        action = glft_expert_action(state)
    elif name == "GLFT-drift":
        action = glft_drift_expert_action(state)
    else:
        raise ValueError(f"Unknown expert name: {name}")

    return np.clip(action, 0.002, 0.250).astype(np.float32)


def choose_expert_action(state: np.ndarray, regime: dict[str, Any]) -> np.ndarray:
    teacher_expert = regime.get("teacher_expert")

    if teacher_expert is None:
        hurst = float(regime.get("hurst", 0.5))
        vol = float(regime.get("vol", 0.02))
        arrival_rate = float(regime.get("arrival_rate", 25.0))
        drift = float(regime.get("drift", 0.0))

        if hurst > 0.5 and abs(drift) > 0:
            teacher_expert = "GLFT-drift"
        elif vol >= 0.25 or arrival_rate <= 25.0:
            teacher_expert = "AS"
        else:
            teacher_expert = "GLFT"

    return expert_action_by_name(str(teacher_expert), state)


def simulate_fills(
    bid_offset: float,
    ask_offset: float,
    buy_arrival: float,
    sell_arrival: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    bid_fill_prob = np.exp(-10.0 * bid_offset) * min(1.0, sell_arrival / 50.0)
    ask_fill_prob = np.exp(-10.0 * ask_offset) * min(1.0, buy_arrival / 50.0)

    bid_fill_prob = float(np.clip(bid_fill_prob, 0.0, 0.98))
    ask_fill_prob = float(np.clip(ask_fill_prob, 0.0, 0.98))

    bid_fill = float(rng.binomial(1, bid_fill_prob))
    ask_fill = float(rng.binomial(1, ask_fill_prob))

    return bid_fill, ask_fill


def generate_dataset(
    regimes: list[dict[str, Any]],
    episodes_per_regime: int,
    rng: np.random.Generator,
    include_pairs: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    path_prices: list[np.ndarray] = []
    path_buys: list[np.ndarray] = []
    path_sells: list[np.ndarray] = []
    path_regime_ids: list[int] = []

    for regime_id, regime in enumerate(regimes):
        for _ in range(episodes_per_regime):
            prices, buys, sells = simulate_market_path(regime, rng, horizon=HORIZON)

            path_prices.append(prices)
            path_buys.append(buys)
            path_sells.append(sells)
            path_regime_ids.append(regime_id)

            inventory = 0.0

            for t in range(HORIZON):
                state = make_state(
                    prices=prices,
                    buy_arrivals=buys,
                    sell_arrivals=sells,
                    t=t,
                    inventory=inventory,
                )

                action = choose_expert_action(state, regime)

                if include_pairs:
                    all_states.append(state)
                    all_actions.append(action)

                bid_fill, ask_fill = simulate_fills(
                    bid_offset=float(action[0]),
                    ask_offset=float(action[1]),
                    buy_arrival=float(buys[t]),
                    sell_arrival=float(sells[t]),
                    rng=rng,
                )

                inventory += bid_fill
                inventory -= ask_fill
                inventory = float(np.clip(inventory, -MAX_INVENTORY, MAX_INVENTORY))

    paths = {
        "prices": np.stack(path_prices).astype(np.float32),
        "buy_arrivals": np.stack(path_buys).astype(np.float32),
        "sell_arrivals": np.stack(path_sells).astype(np.float32),
        "regime_ids": np.asarray(path_regime_ids, dtype=np.int64),
    }

    if include_pairs:
        states = np.stack(all_states).astype(np.float32)
        actions = np.stack(all_actions).astype(np.float32)
    else:
        states = np.empty((0, STATE_DIM), dtype=np.float32)
        actions = np.empty((0, ACTION_DIM), dtype=np.float32)

    return states, actions, paths


def save_paths(path: Path, paths: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        prices=paths["prices"],
        buy_arrivals=paths["buy_arrivals"],
        sell_arrivals=paths["sell_arrivals"],
        regime_ids=paths["regime_ids"],
    )


def write_readme(env_dir: Path) -> None:
    readme = """# FlowHFT-Style Market-Making Data

This directory contains visible data for a CPU-compatible FlowHFT-inspired
market-making benchmark.

## Task

Train or construct a policy that maps a 10-dimensional market state vector to
two positive quote offsets:

- bid_offset
- ask_offset

The policy is evaluated by hidden rollout performance, not imitation loss alone.

## Visible Files

- train_states.npy: training states, shape (N, 10)
- train_actions.npy: analytical expert quote offsets, shape (N, 2)
- public_val_states.npy: public validation states, shape (M, 10)
- public_val_actions.npy: public validation expert actions, shape (M, 2)
- public_val_paths.npz: public market paths for lightweight rollout calibration
- public_val_regimes.npy: episode-level public validation regime IDs
- normalization_stats.npz: state/action summary statistics
- data_card.json: metadata for the visible task

## State Vector

Each state has 10 features:

0. normalized inventory
1. time fraction
2. one-step return
3. recent return mean
4. recent realized volatility
5. recent buy arrival rate divided by 50
6. recent sell arrival rate divided by 50
7. order-flow imbalance
8. price deviation from start
9. remaining time

## Experts

Visible actions are generated from analytical experts:

- Avellaneda-Stoikov style expert
- GLFT style expert
- GLFT with drift style expert

PPO is intentionally removed to keep this benchmark lightweight and
CPU-compatible.

## Hidden Evaluation

Hidden market paths use out-of-sample market settings inspired by the paper:
Hurst/trend settings, drift, volatility, jumps, time step, and Hawkes-style
order arrivals. Hidden regime names and parameters are withheld from the agent.
"""
    (env_dir / "README_DATA.md").write_text(readme, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    env_dir = Path("env_data")
    scoring_dir = Path("scoring_data")

    env_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)

    train_states, train_actions, train_paths = generate_dataset(
        regimes=VISIBLE_REGIMES,
        episodes_per_regime=90,
        rng=rng,
        include_pairs=True,
    )

    public_val_states, public_val_actions, public_val_paths = generate_dataset(
        regimes=VISIBLE_REGIMES,
        episodes_per_regime=15,
        rng=rng,
        include_pairs=True,
    )

    _, _, hidden_paths = generate_dataset(
        regimes=HIDDEN_REGIMES,
        episodes_per_regime=15,
        rng=rng,
        include_pairs=False,
    )

    np.save(env_dir / "train_states.npy", train_states)
    np.save(env_dir / "train_actions.npy", train_actions)
    np.save(env_dir / "public_val_states.npy", public_val_states)
    np.save(env_dir / "public_val_actions.npy", public_val_actions)
    np.save(env_dir / "public_val_regimes.npy", public_val_paths["regime_ids"])

    save_paths(env_dir / "public_val_paths.npz", public_val_paths)
    save_paths(scoring_dir / "hidden_paths.npz", hidden_paths)

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = train_states.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, 1e-6).astype(np.float32)

    action_mean = train_actions.mean(axis=0).astype(np.float32)
    action_std = train_actions.std(axis=0).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6).astype(np.float32)

    np.savez_compressed(
        env_dir / "normalization_stats.npz",
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    data_card = {
        "task_name": "flowhft-cpu-market-making",
        "random_seed": RANDOM_SEED,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
        "max_inventory": MAX_INVENTORY,
        "expert_pool": EXPERT_POOL,
        "ppo_removed": True,
        "paper_style_parameters": {
            "hurst_values": [0.2, 0.5, 0.8],
            "drift_values": [0.0, 0.2],
            "volatility_values": [0.02, 0.25],
            "jump_mean": 0.02,
            "dt_values": [0.01, 0.02],
            "arrival_rates": [25.0, 50.0],
            "self_exciting": 0.7,
            "cross_exciting": 0.3,
        },
        "visible_train": {
            "num_state_action_pairs": int(train_states.shape[0]),
            "num_episodes": int(train_paths["prices"].shape[0]),
            "num_regimes": len(VISIBLE_REGIMES),
            "regime_names": [r["name"] for r in VISIBLE_REGIMES],
        },
        "public_validation": {
            "num_state_action_pairs": int(public_val_states.shape[0]),
            "num_episodes": int(public_val_paths["prices"].shape[0]),
            "num_regimes": len(VISIBLE_REGIMES),
            "regime_names": [r["name"] for r in VISIBLE_REGIMES],
        },
        "hidden_scoring": {
            "num_episodes": int(hidden_paths["prices"].shape[0]),
            "num_regimes": len(HIDDEN_REGIMES),
            "regime_details": "withheld",
            "note": "Hidden regime names and parameters are not exposed to the agent.",
        },
        "state_description": [
            "inventory / max_inventory",
            "time fraction",
            "one-step return",
            "recent return mean",
            "recent realized volatility",
            "recent buy arrivals / 50",
            "recent sell arrivals / 50",
            "order-flow imbalance",
            "price deviation from start",
            "remaining time",
        ],
        "action_description": [
            "bid_offset as fraction of mid-price",
            "ask_offset as fraction of mid-price",
        ],
        "hardware_note": "CPU-compatible. No GPU is required.",
    }

    (env_dir / "data_card.json").write_text(
        json.dumps(data_card, indent=2),
        encoding="utf-8",
    )

    hidden_metadata = {
        "num_hidden_episodes": int(hidden_paths["prices"].shape[0]),
        "num_hidden_regimes": len(HIDDEN_REGIMES),
        "horizon": HORIZON,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "regime_details": "withheld",
        "note": "Hidden regime names and parameters are intentionally withheld.",
    }

    (scoring_dir / "hidden_metadata.json").write_text(
        json.dumps(hidden_metadata, indent=2),
        encoding="utf-8",
    )

    write_readme(env_dir)

    print("Generated FlowHFT-style data.")
    print(f"train_states: {train_states.shape}")
    print(f"train_actions: {train_actions.shape}")
    print(f"public_val_states: {public_val_states.shape}")
    print(f"public_val_actions: {public_val_actions.shape}")
    print(f"public_val_paths episodes: {public_val_paths['prices'].shape[0]}")
    print(f"hidden_paths episodes: {hidden_paths['prices'].shape[0]}")
    print(f"env_data written to: {env_dir.resolve()}")
    print(f"scoring_data written to: {scoring_dir.resolve()}")


if __name__ == "__main__":
    main()
