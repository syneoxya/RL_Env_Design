# scoring_script.py

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


STATE_DIM = 10
ACTION_DIM = 2
MAX_INVENTORY = 20.0

# Hidden rollout settings.
FILL_DECAY = 12.0
ADVERSE_SELECTION_STRENGTH = 10.0
FEE_RATE = 0.0005
LIQUIDATION_PENALTY_RATE = 0.002


def fail(message: str, output_path: str | None = None) -> None:
    result = {
        "score": 0.0,
        "metadata": {
            "error": message,
        },
    }

    if output_path is not None:
        try:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    print(json.dumps(result, indent=2))
    sys.exit(0)


def get_scoring_dir() -> Path:
    try:
        from pm_env.get_data_dir import get_scoring_data_dir
        return Path(get_scoring_data_dir())
    except Exception:
        return Path("/pm_env/scoring_data")


def load_policy(module_path: str, output_path: str | None = None):
    if not os.path.exists(module_path):
        fail(f"Missing policy.py at {module_path}", output_path)

    spec = importlib.util.spec_from_file_location("policy", module_path)

    if spec is None or spec.loader is None:
        fail("Could not import policy.py", output_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "FlowHFTPolicy"):
        fail("policy.py does not define FlowHFTPolicy", output_path)

    model = module.FlowHFTPolicy()

    if not isinstance(model, torch.nn.Module):
        fail("FlowHFTPolicy is not a torch.nn.Module", output_path)

    return model


def load_state_dict(checkpoint_path: str, output_path: str | None = None):
    if not os.path.exists(checkpoint_path):
        fail(f"Missing checkpoint at {checkpoint_path}", output_path)

    try:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        fail(f"Could not load checkpoint: {repr(e)}", output_path)

    if not isinstance(state, dict):
        fail("Checkpoint is not a state_dict dictionary", output_path)

    return state


def validate_model_interface(
    model: torch.nn.Module,
    state: dict,
    output_path: str | None = None,
) -> None:
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameter_count: {param_count}")

    try:
        model.load_state_dict(state)
    except Exception as e:
        fail(f"Checkpoint does not load into FlowHFTPolicy: {repr(e)}", output_path)

    model.eval()

    try:
        with torch.no_grad():
            x = torch.randn(4, STATE_DIM)
            y = model(x)
    except Exception as e:
        fail(f"Model forward pass failed: {repr(e)}", output_path)

    if not isinstance(y, torch.Tensor):
        fail("Model output is not a torch.Tensor", output_path)

    if tuple(y.shape) != (4, ACTION_DIM):
        fail(
            f"Wrong output shape: expected (4, {ACTION_DIM}), got {tuple(y.shape)}",
            output_path,
        )

    if not torch.isfinite(y).all():
        fail("Model output contains NaN or Inf", output_path)

    if not (y > 0).all():
        fail("Model output must contain positive bid/ask offsets", output_path)

    validate_flow_contract(model, output_path)


def validate_flow_contract(
    model: torch.nn.Module,
    output_path: str | None = None,
) -> None:
    """Enforce the architectural requirements that are part of the task contract."""
    buffers = dict(model.named_buffers())
    required_buffers = {
        "state_mean": (STATE_DIM,),
        "state_std": (STATE_DIM,),
        "action_mean": (ACTION_DIM,),
        "action_std": (ACTION_DIM,),
        "alpha": (ACTION_DIM,),
        "beta": (ACTION_DIM,),
    }
    for name, shape in required_buffers.items():
        if name not in buffers:
            fail(f"FlowHFTPolicy must register the {name} buffer", output_path)
        value = buffers[name]
        if tuple(value.shape) != shape:
            fail(
                f"Buffer {name} must have shape {shape}, got {tuple(value.shape)}",
                output_path,
            )
        if not torch.isfinite(value).all():
            fail(f"Buffer {name} contains NaN or Inf", output_path)
    if not (buffers["state_std"] > 0).all() or not (
        buffers["action_std"] > 0
    ).all():
        fail("Normalization standard-deviation buffers must be positive", output_path)

    if not hasattr(model, "vector_field") or not isinstance(
        model.vector_field, torch.nn.Module
    ):
        fail("FlowHFTPolicy must define vector_field as a torch.nn.Module", output_path)
    if not callable(getattr(model, "velocity", None)):
        fail(
            "FlowHFTPolicy must define velocity(action_t, time, normalized_state)",
            output_path,
        )
    if not callable(getattr(model, "raw_action", None)):
        fail("FlowHFTPolicy must define raw_action(state)", output_path)

    calls = []
    hook = model.vector_field.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            x = torch.randn(4, STATE_DIM)
            normalized_state = (x - buffers["state_mean"]) / buffers[
                "state_std"
            ].clamp_min(1e-6)
            action_t = torch.randn(4, ACTION_DIM)
            time = torch.rand(4, 1)
            velocity = model.velocity(action_t, time, normalized_state)
            calls.clear()
            raw = model.raw_action(x)
            output = model(x)
    except Exception as e:
        fail(f"Conditional flow contract check failed: {repr(e)}", output_path)
    finally:
        hook.remove()

    if tuple(velocity.shape) != (4, ACTION_DIM) or not torch.isfinite(velocity).all():
        fail("velocity must return finite values with shape (batch, 2)", output_path)
    if tuple(raw.shape) != (4, ACTION_DIM) or not torch.isfinite(raw).all():
        fail("raw_action must return finite values with shape (batch, 2)", output_path)
    if not calls:
        fail("forward/raw_action must integrate and call the learned vector_field", output_path)

    expected = buffers["alpha"] * raw + buffers["beta"]
    expected = torch.nan_to_num(
        expected, nan=0.002, posinf=0.250, neginf=0.002
    ).clamp(0.002, 0.250)
    if not torch.allclose(output, expected, atol=1e-6, rtol=1e-5):
        fail("forward must apply alpha * raw_action + beta before safe clamping", output_path)


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


def fill_probabilities(
    bid_offset: float,
    ask_offset: float,
    buy_arrival: float,
    sell_arrival: float,
    next_return: float,
) -> tuple[float, float]:
    base_bid_fill = np.exp(-FILL_DECAY * bid_offset) * min(1.0, sell_arrival / 50.0)
    base_ask_fill = np.exp(-FILL_DECAY * ask_offset) * min(1.0, buy_arrival / 50.0)

    # Simple adverse selection model.
    bid_adverse_multiplier = 1.0 + ADVERSE_SELECTION_STRENGTH * max(-next_return, 0.0)
    ask_adverse_multiplier = 1.0 + ADVERSE_SELECTION_STRENGTH * max(next_return, 0.0)

    bid_favorable_discount = 1.0 - 2.0 * max(next_return, 0.0)
    ask_favorable_discount = 1.0 - 2.0 * max(-next_return, 0.0)

    bid_multiplier = np.clip(
        bid_adverse_multiplier * bid_favorable_discount,
        0.35,
        2.00,
    )
    ask_multiplier = np.clip(
        ask_adverse_multiplier * ask_favorable_discount,
        0.35,
        2.00,
    )

    bid_fill_prob = float(np.clip(base_bid_fill * bid_multiplier, 0.0, 0.98))
    ask_fill_prob = float(np.clip(base_ask_fill * ask_multiplier, 0.0, 0.98))

    return bid_fill_prob, ask_fill_prob


def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = peaks - equity_curve
    return float(np.max(drawdowns))


def safe_sharpe(step_returns: np.ndarray) -> float:
    step_returns = np.asarray(step_returns, dtype=np.float64)

    if len(step_returns) < 2:
        return 0.0

    mean = float(np.mean(step_returns))
    std = float(np.std(step_returns))

    if std < 1e-8:
        return 0.0

    return float(mean / std * np.sqrt(len(step_returns)))


# Expert policies used for hidden comparisons.


def as_expert_action(state: np.ndarray) -> np.ndarray:
    """
    Conservative inventory-aware expert.

    This expert should perform well in high-volatility or jumpy regimes.
    """
    inventory = float(state[0])
    realized_vol = abs(float(state[4]))
    remaining_time = float(state[9])

    base_spread = 0.026 + 2.40 * realized_vol + 0.006 * remaining_time
    inventory_penalty = 0.060

    bid = base_spread + inventory_penalty * max(inventory, 0.0)
    ask = base_spread + inventory_penalty * max(-inventory, 0.0)

    return np.array([bid, ask], dtype=np.float32)


def glft_expert_action(state: np.ndarray) -> np.ndarray:
    """
    Liquidity/order-flow expert.

    This expert quotes tighter in liquid lower-volatility markets and skews
    with recent order-flow imbalance.
    """
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
    """
    Drift-aware expert.

    This expert uses short-term return, recent return mean, and order-flow
    imbalance to skew quotes in persistent/trending markets.
    """
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


def constant_action_provider(state: np.ndarray) -> np.ndarray:
    return np.array([0.055, 0.055], dtype=np.float32)


def expert_action_provider(name: str):
    if name == "AS":
        def provider(state: np.ndarray) -> np.ndarray:
            return np.clip(as_expert_action(state), 0.002, 0.250).astype(np.float32)
        return provider

    if name == "GLFT":
        def provider(state: np.ndarray) -> np.ndarray:
            return np.clip(glft_expert_action(state), 0.002, 0.250).astype(np.float32)
        return provider

    if name == "GLFT-drift":
        def provider(state: np.ndarray) -> np.ndarray:
            return np.clip(glft_drift_expert_action(state), 0.002, 0.250).astype(np.float32)
        return provider

    if name == "constant":
        return constant_action_provider

    raise ValueError(f"Unknown expert: {name}")


def model_action_provider(model: torch.nn.Module):
    def provider(state: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():
            action = model(x)

        if not torch.isfinite(action).all():
            raise ValueError("Policy produced NaN or Inf action")

        action_np = action.squeeze(0).cpu().numpy().astype(np.float64)

        if action_np.shape != (2,):
            raise ValueError(f"Policy action has wrong shape: {action_np.shape}")

        if action_np[0] <= 0 or action_np[1] <= 0:
            raise ValueError("Policy produced non-positive quote offset")

        return action_np.astype(np.float32)

    return provider


def rollout_one_episode(
    action_provider,
    prices: np.ndarray,
    buy_arrivals: np.ndarray,
    sell_arrivals: np.ndarray,
) -> dict:
    horizon = len(buy_arrivals)

    cash = 0.0
    inventory = 0.0

    inventories = []
    actions = []
    equity_curve = [0.0]

    for t in range(horizon):
        price = float(prices[t])
        next_price = float(prices[t + 1])
        next_return = (next_price - price) / max(price, 1e-6)

        state = make_state(
            prices=prices,
            buy_arrivals=buy_arrivals,
            sell_arrivals=sell_arrivals,
            t=t,
            inventory=inventory,
        )

        action_np = action_provider(state)

        bid_offset = float(action_np[0])
        ask_offset = float(action_np[1])

        if not np.isfinite(bid_offset) or not np.isfinite(ask_offset):
            raise ValueError("Policy produced non-finite quote offset")

        if bid_offset <= 0 or ask_offset <= 0:
            raise ValueError("Policy produced non-positive quote offset")

        bid_offset = float(np.clip(bid_offset, 0.002, 0.250))
        ask_offset = float(np.clip(ask_offset, 0.002, 0.250))

        bid_fill, ask_fill = fill_probabilities(
            bid_offset=bid_offset,
            ask_offset=ask_offset,
            buy_arrival=float(buy_arrivals[t]),
            sell_arrival=float(sell_arrivals[t]),
            next_return=float(next_return),
        )

        bid_price = price * (1.0 - bid_offset)
        ask_price = price * (1.0 + ask_offset)

        cash -= bid_fill * bid_price
        inventory += bid_fill

        cash += ask_fill * ask_price
        inventory -= ask_fill

        traded_notional = bid_fill * bid_price + ask_fill * ask_price
        cash -= FEE_RATE * traded_notional

        inventory = float(
            np.clip(inventory, -3.0 * MAX_INVENTORY, 3.0 * MAX_INVENTORY)
        )

        equity = cash + inventory * next_price

        inventories.append(inventory)
        actions.append([bid_offset, ask_offset])
        equity_curve.append(equity)

    final_price = float(prices[-1])
    liquidation_penalty = LIQUIDATION_PENALTY_RATE * abs(inventory) * final_price
    final_equity = float(equity_curve[-1] - liquidation_penalty)
    equity_curve[-1] = final_equity

    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity_curve)

    inventories = np.asarray(inventories, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)

    start_price = max(float(prices[0]), 1e-6)

    return {
        "final_pnl": final_equity,
        "normalized_pnl": final_equity / start_price,
        "sharpe": safe_sharpe(step_returns),
        "max_drawdown": max_drawdown(equity_curve) / start_price,
        "avg_abs_inventory": float(np.mean(np.abs(inventories))),
        "max_abs_inventory": float(np.max(np.abs(inventories))),
        "avg_bid_offset": float(np.mean(actions[:, 0])),
        "avg_ask_offset": float(np.mean(actions[:, 1])),
        "std_bid_offset": float(np.std(actions[:, 0])),
        "std_ask_offset": float(np.std(actions[:, 1])),
        "final_inventory": float(inventory),
        "positive_pnl": float(final_equity > 0.0),
    }


def aggregate_metrics(episode_metrics: list[dict], regime_ids: np.ndarray) -> dict:
    keys = [
        "final_pnl",
        "normalized_pnl",
        "sharpe",
        "max_drawdown",
        "avg_abs_inventory",
        "max_abs_inventory",
        "avg_bid_offset",
        "avg_ask_offset",
        "std_bid_offset",
        "std_ask_offset",
        "positive_pnl",
    ]

    aggregate = {}

    for key in keys:
        values = np.array([m[key] for m in episode_metrics], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    regime_summary = {}
    unique_regimes = sorted(set(int(x) for x in regime_ids))

    for regime_id in unique_regimes:
        mask = regime_ids == regime_id
        regime_metrics = [m for m, keep in zip(episode_metrics, mask) if keep]

        regime_summary[str(regime_id)] = {
            "num_episodes": int(len(regime_metrics)),
            "mean_normalized_pnl": float(
                np.mean([m["normalized_pnl"] for m in regime_metrics])
            ),
            "mean_sharpe": float(np.mean([m["sharpe"] for m in regime_metrics])),
            "mean_max_drawdown": float(
                np.mean([m["max_drawdown"] for m in regime_metrics])
            ),
            "mean_avg_abs_inventory": float(
                np.mean([m["avg_abs_inventory"] for m in regime_metrics])
            ),
            "mean_max_abs_inventory": float(
                np.mean([m["max_abs_inventory"] for m in regime_metrics])
            ),
            "positive_pnl_rate": float(
                np.mean([m["positive_pnl"] for m in regime_metrics])
            ),
        }

    aggregate["by_regime"] = regime_summary

    return aggregate


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def mean_metric(aggregate: dict, key: str) -> float:
    return float(aggregate[key]["mean"])


def range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("high must be greater than low")

    return clip01((value - low) / (high - low))


def inverse_range_score(value: float, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError("bad must be greater than good")

    return clip01((bad - value) / (bad - good))


def score_higher_than_experts(
    policy_value: float,
    expert_values: list[float],
    floor: float,
) -> float:
    expert_min = min(expert_values)
    expert_max = max(expert_values)
    denom = max(expert_max - expert_min, floor)

    return clip01((policy_value - expert_min) / denom)


def score_lower_than_experts(
    policy_value: float,
    expert_values: list[float],
    floor: float,
) -> float:
    expert_min = min(expert_values)
    expert_max = max(expert_values)
    denom = max(expert_max - expert_min, floor)

    return clip01((expert_max - policy_value) / denom)


def score_from_metrics(
    policy_aggregate: dict,
    expert_aggregates: dict[str, dict],
) -> tuple[float, dict]:
    regime_ids = sorted(policy_aggregate["by_regime"].keys(), key=int)

    per_regime_scores = {}

    for regime_id in regime_ids:
        # Score each regime separately.
        policy_regime = policy_aggregate["by_regime"][regime_id]
        expert_regimes = [
            aggregate["by_regime"][regime_id]
            for aggregate in expert_aggregates.values()
        ]

        pnl_score = score_higher_than_experts(
            policy_regime["mean_normalized_pnl"],
            [r["mean_normalized_pnl"] for r in expert_regimes],
            floor=0.50,
        )

        sharpe_score = score_higher_than_experts(
            policy_regime["mean_sharpe"],
            [r["mean_sharpe"] for r in expert_regimes],
            floor=1.00,
        )

        drawdown_score = score_lower_than_experts(
            policy_regime["mean_max_drawdown"],
            [r["mean_max_drawdown"] for r in expert_regimes],
            floor=0.05,
        )

        avg_inventory_score = score_lower_than_experts(
            policy_regime["mean_avg_abs_inventory"],
            [r["mean_avg_abs_inventory"] for r in expert_regimes],
            floor=1.00,
        )

        max_inventory_score = score_lower_than_experts(
            policy_regime["mean_max_abs_inventory"],
            [r["mean_max_abs_inventory"] for r in expert_regimes],
            floor=3.00,
        )

        inventory_score = 0.60 * avg_inventory_score + 0.40 * max_inventory_score

        per_regime_scores[regime_id] = {
            "pnl_score": pnl_score,
            "sharpe_score": sharpe_score,
            "drawdown_score": drawdown_score,
            "avg_inventory_score": avg_inventory_score,
            "max_inventory_score": max_inventory_score,
            "inventory_score": inventory_score,
            "positive_pnl_rate": policy_regime["positive_pnl_rate"],
            "policy_mean_normalized_pnl": policy_regime["mean_normalized_pnl"],
            "policy_mean_sharpe": policy_regime["mean_sharpe"],
            "policy_mean_max_drawdown": policy_regime["mean_max_drawdown"],
            "policy_mean_avg_abs_inventory": policy_regime["mean_avg_abs_inventory"],
            "policy_mean_max_abs_inventory": policy_regime["mean_max_abs_inventory"],
        }

    pnl_score = float(np.mean([s["pnl_score"] for s in per_regime_scores.values()]))
    sharpe_score = float(np.mean([s["sharpe_score"] for s in per_regime_scores.values()]))
    drawdown_score = float(
        np.mean([s["drawdown_score"] for s in per_regime_scores.values()])
    )
    inventory_score = float(
        np.mean([s["inventory_score"] for s in per_regime_scores.values()])
    )

    regime_positive_rates = np.array(
        [s["positive_pnl_rate"] for s in per_regime_scores.values()],
        dtype=np.float64,
    )

    regime_pnl_scores = np.array(
        [s["pnl_score"] for s in per_regime_scores.values()],
        dtype=np.float64,
    )

    positive_rate_score = clip01(float(np.mean(regime_positive_rates)))
    expert_competitiveness_score = float(np.mean(regime_pnl_scores))

    # Reward broad regime coverage.
    robustness_score = 0.50 * positive_rate_score + 0.50 * expert_competitiveness_score

    action_std = 0.5 * (
        mean_metric(policy_aggregate, "std_bid_offset")
        + mean_metric(policy_aggregate, "std_ask_offset")
    )

    # Avoid giving full credit to flat quotes.
    adaptivity_score = clip01((action_std - 0.002) / 0.018)

    raw_score = (
        0.30 * pnl_score
        + 0.20 * sharpe_score
        + 0.15 * drawdown_score
        + 0.15 * inventory_score
        + 0.10 * robustness_score
        + 0.10 * adaptivity_score
    )

    final_score = float(np.clip(raw_score, 0.0, 1.0))

    component_scores = {
        "pnl_score": pnl_score,
        "sharpe_score": sharpe_score,
        "drawdown_score": drawdown_score,
        "inventory_score": inventory_score,
        "robustness_score": robustness_score,
        "adaptivity_score": adaptivity_score,
        "positive_rate_score": positive_rate_score,
        "expert_competitiveness_score": expert_competitiveness_score,
        "action_std": float(action_std),
        "raw_score": raw_score,
        "policy_mean_normalized_pnl": mean_metric(policy_aggregate, "normalized_pnl"),
        "policy_mean_sharpe": mean_metric(policy_aggregate, "sharpe"),
        "policy_mean_max_drawdown": mean_metric(policy_aggregate, "max_drawdown"),
        "policy_mean_avg_abs_inventory": mean_metric(policy_aggregate, "avg_abs_inventory"),
        "policy_mean_max_abs_inventory": mean_metric(policy_aggregate, "max_abs_inventory"),
        "per_regime_scores": per_regime_scores,
        "score_notes": {
            "expert_comparison": (
                "Each hidden regime is scored against AS, GLFT, and GLFT-drift "
                "rollouts on that same regime. Higher PnL/Sharpe is better; lower "
                "drawdown/inventory is better."
            ),
            "no_hindsight_oracle": (
                "The scorer does not choose the best expert per episode. It compares "
                "regime-level aggregate metrics against the range spanned by the fixed "
                "expert strategies."
            ),
        },
    }

    return final_score, component_scores


def load_hidden_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scoring_dir = get_scoring_dir()
    hidden_paths_path = scoring_dir / "hidden_paths.npz"

    if not hidden_paths_path.exists():
        fail(f"Missing hidden scoring file: {hidden_paths_path}")

    data = np.load(hidden_paths_path)

    required = ["prices", "buy_arrivals", "sell_arrivals", "regime_ids"]

    for key in required:
        if key not in data:
            fail(f"hidden_paths.npz missing key: {key}")

    prices = data["prices"].astype(np.float32)
    buy_arrivals = data["buy_arrivals"].astype(np.float32)
    sell_arrivals = data["sell_arrivals"].astype(np.float32)
    regime_ids = data["regime_ids"].astype(np.int64)

    return prices, buy_arrivals, sell_arrivals, regime_ids


def rollout_dataset(action_provider, prices, buys, sells, regime_ids) -> tuple[list[dict], dict]:
    episode_metrics = []

    for i in range(prices.shape[0]):
        metrics = rollout_one_episode(
            action_provider=action_provider,
            prices=prices[i],
            buy_arrivals=buys[i],
            sell_arrivals=sells[i],
        )
        episode_metrics.append(metrics)

    aggregate = aggregate_metrics(episode_metrics, regime_ids)

    return episode_metrics, aggregate


def rollout_oracle_expert_dataset(prices, buys, sells, regime_ids) -> tuple[list[dict], dict]:
    # Kept around for debugging old verifier variants.
    candidate_names = ["AS", "GLFT", "GLFT-drift", "constant"]

    best_episode_metrics = []

    for i in range(prices.shape[0]):
        candidates = []

        for candidate_name in candidate_names:
            provider = expert_action_provider(candidate_name)
            metrics = rollout_one_episode(
                action_provider=provider,
                prices=prices[i],
                buy_arrivals=buys[i],
                sell_arrivals=sells[i],
            )
            metrics["selected_expert"] = candidate_name
            candidates.append(metrics)

        best = max(candidates, key=lambda m: m["normalized_pnl"])
        best_episode_metrics.append(best)

    aggregate = aggregate_metrics(best_episode_metrics, regime_ids)

    selected_counts = {}

    for m in best_episode_metrics:
        name = m["selected_expert"]
        selected_counts[name] = selected_counts.get(name, 0) + 1

    aggregate["selected_expert_counts"] = selected_counts

    return best_episode_metrics, aggregate


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python scoring_script.py "
            "<policy.py> <flowhft_policy.pt> <output_results.txt>"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_path = sys.argv[3]

    model = load_policy(module_path, output_path)
    state = load_state_dict(checkpoint_path, output_path)
    validate_model_interface(model, state, output_path)

    model.eval()

    prices, buys, sells, regime_ids = load_hidden_paths()

    if prices.ndim != 2:
        fail(
            f"Expected prices shape (episodes, horizon+1), got {prices.shape}",
            output_path,
        )

    try:
        _, policy_aggregate = rollout_dataset(
            action_provider=model_action_provider(model),
            prices=prices,
            buys=buys,
            sells=sells,
            regime_ids=regime_ids,
        )

        expert_aggregates = {}

        for expert_name in ["AS", "GLFT", "GLFT-drift"]:
            _, expert_aggregate = rollout_dataset(
                action_provider=expert_action_provider(expert_name),
                prices=prices,
                buys=buys,
                sells=sells,
                regime_ids=regime_ids,
            )
            expert_aggregates[expert_name] = expert_aggregate

    except Exception as e:
        fail(f"Rollout failed: {repr(e)}", output_path)

    final_score, component_scores = score_from_metrics(
        policy_aggregate=policy_aggregate,
        expert_aggregates=expert_aggregates,
    )

    metadata = {
        "component_scores": component_scores,
        "policy_aggregate_metrics": policy_aggregate,
        "expert_baseline_summaries": {
            expert_name: {
                "mean_normalized_pnl": aggregate["normalized_pnl"]["mean"],
                "mean_sharpe": aggregate["sharpe"]["mean"],
                "mean_drawdown": aggregate["max_drawdown"]["mean"],
                "mean_avg_abs_inventory": aggregate["avg_abs_inventory"]["mean"],
                "mean_max_abs_inventory": aggregate["max_abs_inventory"]["mean"],
                "by_regime": aggregate["by_regime"],
            }
            for expert_name, aggregate in expert_aggregates.items()
        },
        "hidden_metadata": {
            "num_hidden_episodes": int(prices.shape[0]),
            "num_hidden_regimes": int(len(set(regime_ids.tolist()))),
            "horizon": int(prices.shape[1] - 1),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "note": "Hidden regime names and parameters are withheld from the agent.",
        },
        "final_score_formula": (
            "0.30 regime-level PnL versus AS/GLFT/GLFT-drift + "
            "0.20 regime-level Sharpe versus AS/GLFT/GLFT-drift + "
            "0.15 regime-level drawdown control versus AS/GLFT/GLFT-drift + "
            "0.15 regime-level inventory control versus AS/GLFT/GLFT-drift + "
            "0.10 robustness score + "
            "0.10 action adaptivity score"
        ),
        "scoring_notes": (
            "The policy is rolled out on hidden market paths with transaction costs "
            "and adverse-selection fill dynamics. The score compares the submitted "
            "policy against AS, GLFT, and GLFT-drift experts on the same hidden regimes. "
            "It uses regime-level aggregate metrics rather than a hindsight per-episode "
            "oracle. Hidden regime details are not exposed in the output metadata."
        ),
    }

    result = {
        "score": final_score,
        "metadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Short summary before the full JSON.
    summary_keys = [
        "pnl_score",
        "sharpe_score",
        "drawdown_score",
        "inventory_score",
        "robustness_score",
        "adaptivity_score",
    ]
    summary = {
        key: component_scores[key]
        for key in summary_keys
    }

    print(f"final_score: {final_score:.6f}")
    print(f"component_scores: {json.dumps(summary, sort_keys=True)}")
    print(
        "policy_metrics: "
        + json.dumps(
            {
                "mean_normalized_pnl": component_scores["policy_mean_normalized_pnl"],
                "mean_sharpe": component_scores["policy_mean_sharpe"],
                "mean_max_drawdown": component_scores["policy_mean_max_drawdown"],
                "mean_avg_abs_inventory": component_scores[
                    "policy_mean_avg_abs_inventory"
                ],
                "mean_max_abs_inventory": component_scores[
                    "policy_mean_max_abs_inventory"
                ],
                "action_std": component_scores["action_std"],
            },
            sort_keys=True,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
