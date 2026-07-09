from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="flowhft_mpl_"))

import matplotlib.pyplot as plt
import numpy as np


COMPONENT_KEYS = [
    "pnl_score",
    "sharpe_score",
    "drawdown_score",
    "inventory_score",
    "robustness_score",
    "adaptivity_score",
]

COMPONENT_LABELS = {
    "pnl_score": "PnL",
    "sharpe_score": "Sharpe",
    "drawdown_score": "Drawdown",
    "inventory_score": "Inventory",
    "robustness_score": "Robustness",
    "adaptivity_score": "Adaptivity",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def style_axes(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_training_history(summary: dict[str, Any], output_dir: Path) -> Path | None:
    history = summary.get("history") or []
    if not history:
        return None

    steps = [row["step"] for row in history]
    visible_scores = [row["visible_score"] for row in history]
    reward_means = [row["reward_mean"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, visible_scores, marker="o", linewidth=2)
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylim(0, 1)
    style_axes(axes[0], "Visible Rollout Score", "Score")

    axes[1].plot(steps, reward_means, marker="o", linewidth=2, color="#0f766e")
    axes[1].set_xlabel("GRPO step")
    style_axes(axes[1], "Group Reward Mean", "Reward")

    fig.suptitle("GRPO Training Progress", fontsize=15, y=1.03)
    return savefig(output_dir, "grpo_training_progress.png")


def plot_score_components(results: dict[str, Any], output_dir: Path) -> Path:
    components = results["metadata"]["component_scores"]
    values = [float(components[key]) for key in COMPONENT_KEYS]
    labels = [COMPONENT_LABELS[key] for key in COMPONENT_KEYS]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#ca8a04", "#dc2626", "#475569"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    style_axes(ax, f"Verifier Score Components: final={results['score']:.3f}", "Score")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    return savefig(output_dir, "verifier_score_components.png")


def plot_per_regime_scores(results: dict[str, Any], output_dir: Path) -> Path:
    per_regime = results["metadata"]["component_scores"]["per_regime_scores"]
    regime_ids = sorted(per_regime.keys(), key=int)
    x = np.arange(len(regime_ids))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ("pnl_score", "PnL", "#2563eb"),
        ("sharpe_score", "Sharpe", "#0f766e"),
        ("drawdown_score", "Drawdown", "#7c3aed"),
        ("inventory_score", "Inventory", "#ca8a04"),
    ]

    for i, (key, label, color) in enumerate(series):
        values = [float(per_regime[regime_id][key]) for regime_id in regime_ids]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"R{regime_id}" for regime_id in regime_ids])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax, "Per-Regime Hidden Verifier Scores", "Score")
    fig.subplots_adjust(top=0.78)
    return savefig(output_dir, "per_regime_scores.png")


def plot_policy_vs_experts(results: dict[str, Any], output_dir: Path) -> Path:
    policy = results["metadata"]["policy_aggregate_metrics"]
    experts = results["metadata"]["expert_baseline_summaries"]

    metric_specs = [
        ("normalized_pnl", "mean_normalized_pnl", "Normalized PnL"),
        ("sharpe", "mean_sharpe", "Sharpe"),
        ("max_drawdown", "mean_drawdown", "Drawdown"),
        ("avg_abs_inventory", "mean_avg_abs_inventory", "Avg |Inventory|"),
    ]
    names = ["Policy", *experts.keys()]
    x = np.arange(len(metric_specs))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#2563eb", "#64748b", "#94a3b8", "#cbd5e1"]

    for offset_idx, name in enumerate(names):
        values = []
        for policy_key, expert_key, _ in metric_specs:
            if name == "Policy":
                values.append(float(policy[policy_key]["mean"]))
            else:
                values.append(float(experts[name][expert_key]))
        ax.bar(
            x + (offset_idx - 1.5) * width,
            values,
            width=width,
            label=name,
            color=colors[offset_idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in metric_specs])
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax, "Policy vs Expert Baselines", "Mean Metric")
    fig.subplots_adjust(top=0.78)
    return savefig(output_dir, "policy_vs_experts.png")


def plot_regime_pnl_inventory(results: dict[str, Any], output_dir: Path) -> Path:
    per_regime = results["metadata"]["component_scores"]["per_regime_scores"]
    regime_ids = sorted(per_regime.keys(), key=int)
    pnl = [float(per_regime[regime_id]["policy_mean_normalized_pnl"]) for regime_id in regime_ids]
    inventory = [
        float(per_regime[regime_id]["policy_mean_avg_abs_inventory"])
        for regime_id in regime_ids
    ]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(regime_ids))
    bars = ax1.bar(x, pnl, color="#2563eb", alpha=0.85, label="Normalized PnL")
    ax1.axhline(0, color="#334155", linewidth=1)
    ax1.set_ylabel("Normalized PnL")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"R{regime_id}" for regime_id in regime_ids])
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, inventory, marker="o", color="#dc2626", linewidth=2, label="Avg |Inventory|")
    ax2.set_ylabel("Avg |Inventory|")
    ax2.spines["top"].set_visible(False)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    ax1.bar_label(bars, labels=[f"{v:.1f}" for v in pnl], padding=3, fontsize=8)
    ax1.set_title("Regime PnL and Inventory Risk", fontsize=13, pad=10)
    return savefig(output_dir, "regime_pnl_inventory.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create FlowHFT GRPO result graphs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("env_data/grpo_training_summary.json"),
        help="Path to grpo_training_summary.json.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("out/grpo_results.json"),
        help="Path to verifier results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures"),
        help="Directory where PNG figures are written.",
    )
    args = parser.parse_args()

    summary = load_json(args.summary)
    results = load_json(args.results)

    written: list[Path] = []
    if summary is not None:
        path = plot_training_history(summary, args.output_dir)
        if path is not None:
            written.append(path)

    if results is None:
        raise FileNotFoundError(f"Missing verifier results JSON: {args.results}")

    written.extend(
        [
            plot_score_components(results, args.output_dir),
            plot_per_regime_scores(results, args.output_dir),
            plot_policy_vs_experts(results, args.output_dir),
            plot_regime_pnl_inventory(results, args.output_dir),
        ]
    )

    print("Wrote figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
