from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER_DIR = ROOT.parent / "paper_revision_full"

METHOD_LABELS = {
    "stochastic_planning": "Scenario\nplanning",
    "two_stage_sp": "Two-stage\nSP",
    "plain_ppo": "Plain\nPPO",
    "transformer_ppo": "Transformer\nPPO",
    "gat_ppo": "GAT\nPPO",
    "gnn_ppo": "GNN\nPPO",
    "hybrid_gnn_ppo": "Hybrid\nGNN-PPO",
}

METHOD_ORDER = [
    "stochastic_planning",
    "two_stage_sp",
    "plain_ppo",
    "transformer_ppo",
    "gat_ppo",
    "gnn_ppo",
    "hybrid_gnn_ppo",
]


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )


def _colors(methods: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.colormaps.get_cmap("tab10")
    return {method: cmap(i % 10) for i, method in enumerate(methods)}


def save_main_comparison() -> None:
    _style()
    frame = pd.read_csv(RESULTS / "comparison" / "baseline_vs_ppo_episode_metrics.csv")
    methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [_colors(methods)[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    specs = [
        ("total_profit", "Total Profit", "Million USD", 1e6),
        ("service_rate", "Service Rate", "Ratio", 1.0),
    ]
    for axis, (column, title, ylabel, scale) in zip(axes, specs):
        means = frame.groupby("method")[column].mean().reindex(methods)
        stds = frame.groupby("method")[column].std().reindex(methods).fillna(0.0)
        axis.bar(np.arange(len(methods)), means.to_numpy() / scale, yerr=stds.to_numpy() / scale, capsize=4, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(labels, rotation=0)
        axis.grid(axis="y", alpha=0.3)
        if column == "service_rate":
            axis.set_ylim(0.84, 0.98)
    fig.tight_layout()
    out = RESULTS / "comparison" / "baseline_vs_ppo_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "baseline_vs_ppo_overview_latest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _level_labels(factor: str, levels: list[float]) -> list[str]:
    if factor == "demand_scale":
        return [f"{int(x)}" for x in levels]
    if factor == "joint_supply_transport_risk":
        return [f"{x:.1f}x\n+{x:.1f}x" for x in levels]
    return [f"{x:.1f}x" for x in levels]


def save_sensitivity(factor: str, title: str, output_name: str) -> None:
    _style()
    frame = pd.read_csv(RESULTS / "sensitivity" / factor / "episode_metrics.csv")
    summary = frame.groupby(["level", "method"], as_index=False).agg(
        total_profit_mean=("total_profit", "mean"),
        service_rate_mean=("service_rate", "mean"),
        weekly_profit_cvar20_mean=("weekly_profit_cvar20", "mean"),
    )
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
    levels = sorted(float(x) for x in summary["level"].unique())
    x = np.arange(len(levels), dtype=float)
    colors = _colors(methods)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    specs = [
        ("total_profit_mean", "Profit", "Million USD", 1e6),
        ("service_rate_mean", "Service", "Ratio", 1.0),
        ("weekly_profit_cvar20_mean", "CVaR20", "Million USD", 1e6),
    ]
    for axis, (column, panel_title, ylabel, scale) in zip(axes, specs):
        for method in methods:
            subset = summary[summary["method"] == method].sort_values("level")
            axis.plot(x, subset[column].to_numpy() / scale, marker="o", linewidth=2.2, markersize=5, color=colors[method], label=METHOD_LABELS[method].replace("\n", " "))
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(_level_labels(factor, levels))
        axis.grid(alpha=0.3)
        if column == "service_rate_mean":
            axis.set_ylim(0.82, 0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    out = RESULTS / "sensitivity" / factor / output_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / output_name.replace(".png", "_latest.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_policy_trajectories() -> None:
    _style()
    frame = pd.read_csv(RESULTS / "policy_behavior" / "policy_weekly_trace.csv")
    methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    colors = _colors(methods)
    specs = [
        ("safety_stock_weeks", "Safety Stock", "Weeks"),
        ("backup_route_ratio", "Backup Route", "Ratio"),
        ("futures_hedge_ratio", "Futures Hedge", "Ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    for axis, (column, title, ylabel) in zip(axes, specs):
        for method in methods:
            subset = frame[frame["method"] == method].groupby("week", as_index=False)[column].mean()
            axis.plot(subset["week"], subset[column], linewidth=2.2, color=colors[method], label=METHOD_LABELS[method].replace("\n", " "))
        axis.set_title(title)
        axis.set_xlabel("Week")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    out = RESULTS / "policy_behavior" / "policy_control_trajectories.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "policy_control_trajectories_latest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_main_comparison()
    save_sensitivity("demand_scale", "Demand Scale Sensitivity", "demand_scale_overview.png")
    save_sensitivity("supply_risk", "Supply Risk Sensitivity", "supply_risk_overview.png")
    save_sensitivity("transport_risk", "Transport Risk Sensitivity", "transport_risk_overview.png")
    save_sensitivity("joint_supply_transport_risk", "Joint Supply--Transport Risk Stress Test", "joint_supply_transport_risk_overview.png")
    save_policy_trajectories()


if __name__ == "__main__":
    main()
