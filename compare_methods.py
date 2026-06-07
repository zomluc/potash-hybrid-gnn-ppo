from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from decision_strategy import base_policy, optimize_policy_for_week
from ppo_trainer import load_checkpoint
from rl_environment import PotashSupplyChainEnv
from rl_models import observation_to_torch
from simulation_types import ModelConfig
from supply_chain_dynamics import simulate_horizon
from two_stage_sp import apply_two_stage_plan_one_week, solve_two_stage_sp_for_week


DEFAULT_COMPARISON_SEEDS = [
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1001,
    1102,
    1203,
    1304,
    1405,
    1506,
    1607,
    1708,
    1809,
    1901,
    2002,
]


@dataclass
class PPOComparisonSpec:
    checkpoint_path: str
    model_type: str
    method_label: str | None = None
    gnn_use_temporal_history: bool = True
    gnn_use_attention_pooling: bool = True
    gnn_use_risk_film: bool = True
    gnn_use_edge_enhancement: bool = False


@dataclass
class ComparisonConfig:
    seeds: List[int]
    output_dir: str = "results/comparison"
    device: str = "cpu"
    ppo_specs: List[PPOComparisonSpec] | None = None
    include_baseline: bool = True
    include_two_stage_sp: bool = True
    baseline_planning_scenarios: int | None = None
    write_partial_results: bool = True
    progress: bool = True


def default_comparison_config() -> ComparisonConfig:
    return ComparisonConfig(
        seeds=list(DEFAULT_COMPARISON_SEEDS),
        ppo_specs=[
            PPOComparisonSpec(
                checkpoint_path="results/plain_ppo/ppo_best.pt",
                model_type="mlp",
                method_label="plain_ppo",
            ),
            PPOComparisonSpec(
                checkpoint_path="results/transformer_ppo/ppo_best.pt",
                model_type="transformer",
                method_label="transformer_ppo",
            ),
            PPOComparisonSpec(
                checkpoint_path="results/gat_ppo/ppo_best.pt",
                model_type="gat",
                method_label="gat_ppo",
            ),
            PPOComparisonSpec(
                checkpoint_path="results/gnn_ppo/ppo_best.pt",
                model_type="gnn",
                method_label="gnn_ppo",
            ),
            PPOComparisonSpec(
                checkpoint_path="results/ppo/ppo_best.pt",
                model_type="hybrid",
                method_label="hybrid_gnn_ppo",
            ),
        ],
    )


def resolve_method_name(model_type: str) -> str:
    if model_type == "mlp":
        return "plain_ppo"
    if model_type == "transformer":
        return "transformer_ppo"
    if model_type == "gat":
        return "gat_ppo"
    if model_type == "hybrid":
        return "hybrid_gnn_ppo"
    return "gnn_ppo"


def ensure_checkpoint(config: ComparisonConfig, spec: PPOComparisonSpec) -> Path:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Please retrain this method with the unified training entrypoints before running compare_methods.py."
        )
    return checkpoint_path


def _print_progress(config: ComparisonConfig, payload: dict[str, Any]) -> None:
    if config.progress:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def _empty_episode_stats(method: str, seed: int) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "total_profit": 0.0,
        "service_rate": 0.0,
        "total_shortage": 0.0,
        "avg_supply_risk": 0.0,
        "avg_transport_risk": 0.0,
        "avg_price_risk": 0.0,
        "risk_score": 0.0,
        "weekly_profit_cvar20": 0.0,
        "mean_weekly_reward": 0.0,
    }


def _finalize_episode_stats(stats: dict[str, Any], weekly_profit: List[float], weekly_reward: List[float], steps: int) -> dict[str, Any]:
    steps = max(steps, 1)
    stats["service_rate"] = 1.0 - stats["total_shortage"] / max(stats.get("total_demand", 1.0), 1.0)
    stats["avg_supply_risk"] /= steps
    stats["avg_transport_risk"] /= steps
    stats["avg_price_risk"] /= steps
    stats["risk_score"] = (
        0.40 * stats["avg_supply_risk"]
        + 0.40 * stats["avg_transport_risk"]
        + 0.20 * stats["avg_price_risk"]
    )
    if weekly_profit:
        ordered = sorted(weekly_profit)
        count = max(1, int(np.ceil(len(ordered) * 0.2)))
        stats["weekly_profit_cvar20"] = float(np.mean(ordered[:count]))
    stats["mean_weekly_reward"] = float(np.mean(weekly_reward)) if weekly_reward else 0.0
    stats.pop("total_demand", None)
    return stats


def evaluate_baseline_episode(seed: int, planning_scenarios: int | None = None) -> dict[str, Any]:
    config = ModelConfig(planning_scenarios=int(planning_scenarios)) if planning_scenarios is not None else None
    env = PotashSupplyChainEnv(config=config)
    observation, _ = env.reset(seed=seed)
    del observation
    stats = _empty_episode_stats("stochastic_planning", seed)
    current_policy = base_policy()
    done = False
    weekly_profit = []
    weekly_reward = []
    steps = 0

    while not done:
        assert env.state is not None
        assert env.market is not None
        current_policy = optimize_policy_for_week(
            env.state,
            env.week,
            current_policy,
            env.config,
            env.suppliers,
            env.routes,
            env.market,
            env.base_demand,
            simulate_horizon,
        )
        action = env.encode_policy(current_policy)
        _, reward, terminated, truncated, info = env.step(action)
        metrics = info["metrics"]
        risk = info["risk_summary"]

        stats["total_profit"] += float(metrics["weekly_profit"])
        stats["total_shortage"] += float(metrics["shortage"])
        stats["total_demand"] = stats.get("total_demand", 0.0) + float(metrics["demand"])
        stats["avg_supply_risk"] += float(risk["current_supply_risk"])
        stats["avg_transport_risk"] += float(risk["current_transport_risk"])
        stats["avg_price_risk"] += float(risk["current_price_risk"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        weekly_reward.append(float(reward))
        steps += 1
        done = terminated or truncated

    return _finalize_episode_stats(stats, weekly_profit, weekly_reward, steps)


def evaluate_two_stage_sp_episode(seed: int, planning_scenarios: int | None = None) -> dict[str, Any]:
    config = ModelConfig(planning_scenarios=int(planning_scenarios)) if planning_scenarios is not None else None
    env = PotashSupplyChainEnv(config=config)
    observation, _ = env.reset(seed=seed)
    del observation
    stats = _empty_episode_stats("two_stage_sp", seed)
    done = False
    weekly_profit: list[float] = []
    weekly_reward: list[float] = []
    steps = 0

    while not done:
        assert env.state is not None
        assert env.market is not None
        plan = solve_two_stage_sp_for_week(
            env.state,
            env.week,
            env.config,
            env.suppliers,
            env.routes,
            env.market,
            env.base_demand,
        )
        metrics = apply_two_stage_plan_one_week(
            env.state,
            env.week,
            plan,
            env.config,
            env.suppliers,
            env.routes,
            env.market,
            env.base_demand,
        )
        reward = float(metrics["weekly_profit"] / env.reward_scale)
        env.week += 1
        terminated = env.week >= env.config.horizon_weeks
        truncated = False
        info = {
            "week": env.week,
            "metrics": metrics,
            "risk_summary": env._risk_info_dict(),
        }
        env.last_info = info
        risk = info["risk_summary"]

        stats["total_profit"] += float(metrics["weekly_profit"])
        stats["total_shortage"] += float(metrics["shortage"])
        stats["total_demand"] = stats.get("total_demand", 0.0) + float(metrics["demand"])
        stats["avg_supply_risk"] += float(risk["current_supply_risk"])
        stats["avg_transport_risk"] += float(risk["current_transport_risk"])
        stats["avg_price_risk"] += float(risk["current_price_risk"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        weekly_reward.append(float(reward))
        steps += 1
        done = terminated or truncated

    return _finalize_episode_stats(stats, weekly_profit, weekly_reward, steps)


def evaluate_ppo_episode(
    model,
    seed: int,
    device: str = "cpu",
    method_name: str = "gnn_ppo",
) -> dict[str, Any]:
    env = PotashSupplyChainEnv()
    observation, _ = env.reset(seed=seed)
    stats = _empty_episode_stats(method_name, seed)
    done = False
    weekly_profit = []
    weekly_reward = []
    steps = 0

    while not done:
        obs_t = observation_to_torch(observation, device=device)
        with torch.no_grad():
            action, _, _ = model.act(obs_t, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        metrics = info["metrics"]
        risk = info["risk_summary"]

        stats["total_profit"] += float(metrics["weekly_profit"])
        stats["total_shortage"] += float(metrics["shortage"])
        stats["total_demand"] = stats.get("total_demand", 0.0) + float(metrics["demand"])
        stats["avg_supply_risk"] += float(risk["current_supply_risk"])
        stats["avg_transport_risk"] += float(risk["current_transport_risk"])
        stats["avg_price_risk"] += float(risk["current_price_risk"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        weekly_reward.append(float(reward))
        steps += 1
        done = terminated or truncated

    return _finalize_episode_stats(stats, weekly_profit, weekly_reward, steps)


def summarize_results(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby("method").agg(
        n=("total_profit", "count"),
        total_profit_mean=("total_profit", "mean"),
        total_profit_std=("total_profit", "std"),
        service_rate_mean=("service_rate", "mean"),
        service_rate_std=("service_rate", "std"),
        total_shortage_mean=("total_shortage", "mean"),
        risk_score_mean=("risk_score", "mean"),
        weekly_profit_cvar20_mean=("weekly_profit_cvar20", "mean"),
        weekly_profit_cvar20_std=("weekly_profit_cvar20", "std"),
        mean_weekly_reward_mean=("mean_weekly_reward", "mean"),
    )
    for metric in ["total_profit", "service_rate", "weekly_profit_cvar20"]:
        sem = grouped[f"{metric}_std"] / grouped["n"].clip(lower=1).pow(0.5)
        grouped[f"{metric}_ci95_low"] = grouped[f"{metric}_mean"] - 1.96 * sem
        grouped[f"{metric}_ci95_high"] = grouped[f"{metric}_mean"] + 1.96 * sem
    return grouped.reset_index().to_dict(orient="records")


def _bootstrap_mean_diff_ci(values_a: np.ndarray, values_b: np.ndarray, seed: int = 20260525) -> dict[str, float]:
    if len(values_a) == 0 or len(values_a) != len(values_b):
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = values_a - values_b
    draws = rng.choice(diffs, size=(5000, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def _normal_two_sided_pvalue(statistic: float) -> float:
    return float(math.erfc(abs(float(statistic)) / math.sqrt(2.0)))


def summarize_pairwise_statistics(frame: pd.DataFrame, metric: str = "total_profit") -> list[dict[str, Any]]:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        scipy_stats = None

    methods = sorted(frame["method"].unique())
    comparisons: list[dict[str, Any]] = []
    for left_index, method_a in enumerate(methods):
        for method_b in methods[left_index + 1:]:
            paired = frame[frame["method"].isin([method_a, method_b])].pivot(index="seed", columns="method", values=metric).dropna()
            if paired.empty:
                continue
            values_a = paired[method_a].to_numpy(dtype=float)
            values_b = paired[method_b].to_numpy(dtype=float)
            diff = values_a - values_b
            mean_diff = float(diff.mean())
            diff_std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            ci = _bootstrap_mean_diff_ci(values_a, values_b)
            if scipy_stats is not None and len(diff) > 1:
                t_result = scipy_stats.ttest_rel(values_a, values_b)
                try:
                    wilcoxon_result = scipy_stats.wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
                    wilcoxon_pvalue = float(wilcoxon_result.pvalue)
                except ValueError:
                    wilcoxon_pvalue = float("nan")
                t_statistic = float(t_result.statistic)
                t_pvalue = float(t_result.pvalue)
            elif len(diff) > 1 and diff_std > 0.0:
                t_statistic = mean_diff / (diff_std / math.sqrt(len(diff)))
                t_pvalue = _normal_two_sided_pvalue(t_statistic)
                wilcoxon_pvalue = float("nan")
            else:
                t_statistic = float("nan")
                t_pvalue = float("nan")
                wilcoxon_pvalue = float("nan")
            comparisons.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_paired_seeds": int(len(paired)),
                    "mean_difference_a_minus_b": mean_diff,
                    "std_difference": diff_std,
                    "bootstrap_ci95_low": ci["low"],
                    "bootstrap_ci95_high": ci["high"],
                    "paired_t_statistic": t_statistic,
                    "paired_t_pvalue": t_pvalue,
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                }
            )
    return comparisons


def save_comparison_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    metrics = [
        ("total_profit", "Total Profit", "USD"),
        ("service_rate", "Service Rate", "Ratio"),
    ]
    methods = sorted(frame["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes = np.atleast_1d(axes)
    color_map = plt.cm.get_cmap("tab10", max(1, len(methods)))
    colors = [color_map(index) for index in range(len(methods))]
    for axis, (column, title, ylabel) in zip(axes, metrics):
        means = frame.groupby("method")[column].mean().reindex(methods)
        stds = frame.groupby("method")[column].std().reindex(methods).fillna(0.0)
        plot_values = means.values / 1e6 if column == "total_profit" else means.values
        plot_errors = stds.values / 1e6 if column == "total_profit" else stds.values
        axis.bar(methods, plot_values, yerr=plot_errors, capsize=4, color=colors)
        axis.set_title(title)
        axis.set_ylabel("Million USD" if column == "total_profit" else ylabel)
        axis.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_ppo_overview.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric in zip(axes, ["total_profit", "service_rate"]):
        for method in methods:
            subset = frame[frame["method"] == method].sort_values("seed")
            values = subset[metric] / 1e6 if metric == "total_profit" else subset[metric]
            axis.plot(subset["seed"], values, marker="o", linewidth=2, label=method)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Seed")
        axis.set_ylabel("Million USD" if metric == "total_profit" else "Ratio")
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_ppo_by_seed.png", dpi=220)
    plt.close(fig)


def run_comparison(config: ComparisonConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or default_comparison_config()
    ppo_specs = config.ppo_specs or []
    models: list[tuple[Any, Path, str, str]] = []
    output_dir = Path(config.output_dir)
    if config.write_partial_results:
        output_dir.mkdir(parents=True, exist_ok=True)
    for spec in ppo_specs:
        _print_progress(config, {"event": "load_checkpoint_start", "checkpoint_path": spec.checkpoint_path})
        checkpoint_path = ensure_checkpoint(config, spec)
        model, payload, _ = load_checkpoint(checkpoint_path, device=config.device)
        payload_model_type = payload.get("config", {}).get("model_type", spec.model_type)
        method_name = spec.method_label or resolve_method_name(payload_model_type)
        del payload
        model.eval()
        models.append((model, checkpoint_path, payload_model_type, method_name))
        _print_progress(config, {"event": "load_checkpoint_complete", "method": method_name, "checkpoint_path": str(checkpoint_path)})

    records = []
    baseline_count = (1 if config.include_baseline else 0) + (1 if config.include_two_stage_sp else 0)
    total_runs = len(config.seeds) * (len(models) + baseline_count)
    completed_runs = 0
    _print_progress(config, {"event": "comparison_start", "seeds": config.seeds, "total_runs": total_runs})
    for seed_index, seed in enumerate(config.seeds, start=1):
        _print_progress(config, {"event": "seed_start", "seed": seed, "seed_index": seed_index, "seed_count": len(config.seeds)})
        if config.include_baseline:
            _print_progress(config, {"event": "method_start", "method": "stochastic_planning", "seed": seed})
            records.append(evaluate_baseline_episode(seed, planning_scenarios=config.baseline_planning_scenarios))
            completed_runs += 1
            _print_progress(config, {"event": "method_complete", "method": "stochastic_planning", "seed": seed, "completed_runs": completed_runs, "total_runs": total_runs})
        if config.include_two_stage_sp:
            _print_progress(config, {"event": "method_start", "method": "two_stage_sp", "seed": seed})
            records.append(evaluate_two_stage_sp_episode(seed, planning_scenarios=config.baseline_planning_scenarios))
            completed_runs += 1
            _print_progress(config, {"event": "method_complete", "method": "two_stage_sp", "seed": seed, "completed_runs": completed_runs, "total_runs": total_runs})
        for model, _, _, method_name in models:
            _print_progress(config, {"event": "method_start", "method": method_name, "seed": seed})
            records.append(
                evaluate_ppo_episode(
                    model,
                    seed,
                    device=config.device,
                    method_name=method_name,
                )
            )
            completed_runs += 1
            _print_progress(config, {"event": "method_complete", "method": method_name, "seed": seed, "completed_runs": completed_runs, "total_runs": total_runs})
        if config.write_partial_results and records:
            pd.DataFrame(records).to_csv(output_dir / "baseline_vs_ppo_episode_metrics.partial.csv", index=False)
        _print_progress(config, {"event": "seed_complete", "seed": seed, "seed_index": seed_index, "seed_count": len(config.seeds)})

    frame = pd.DataFrame(records)
    summary = {
        "checkpoint_paths": [
            {
                "method": method_name,
                "checkpoint_path": str(checkpoint_path),
                "ppo_model_type": payload_model_type,
            }
            for _, checkpoint_path, payload_model_type, method_name in models
        ],
        "seeds": config.seeds,
        "include_baseline": config.include_baseline,
        "include_two_stage_sp": config.include_two_stage_sp,
        "baseline_planning_scenarios": config.baseline_planning_scenarios,
        "summary_by_method": summarize_results(frame),
        "pairwise_statistics": {
            metric: summarize_pairwise_statistics(frame, metric=metric)
            for metric in ["total_profit", "service_rate", "weekly_profit_cvar20"]
        },
    }

    for model, _, _, _ in models:
        del model
    _print_progress(config, {"event": "comparison_complete", "completed_runs": completed_runs, "total_runs": total_runs})
    return frame, summary


def save_comparison_outputs(frame: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "baseline_vs_ppo_episode_metrics.csv", index=False)
    with (output_dir / "baseline_vs_ppo_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    save_comparison_plots(frame, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stochastic planning and PPO methods on common test seeds.")
    parser.add_argument("--seed-count", type=int, default=None, help="Use the first N default comparison seeds.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seed list. Overrides --seed-count.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for comparison results.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--skip-baseline", action="store_true", help="Evaluate PPO checkpoints only; useful for quick diagnostics.")
    parser.add_argument("--skip-two-stage-sp", action="store_true", help="Disable the rolling two-stage stochastic programming baseline.")
    parser.add_argument(
        "--baseline-planning-scenarios",
        type=int,
        default=None,
        help="Override stochastic-planning lookahead scenario count. Smaller values run much faster.",
    )
    parser.add_argument("--no-partial", action="store_true", help="Disable partial CSV writing after each seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable JSON progress logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_comparison_config()
    if args.seeds:
        config.seeds = list(args.seeds)
    elif args.seed_count is not None:
        config.seeds = config.seeds[: max(0, int(args.seed_count))]
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    config.include_baseline = not bool(args.skip_baseline)
    config.include_two_stage_sp = (not bool(args.skip_baseline)) and (not bool(args.skip_two_stage_sp))
    config.baseline_planning_scenarios = args.baseline_planning_scenarios
    config.write_partial_results = not bool(args.no_partial)
    config.progress = not bool(args.quiet)
    frame, summary = run_comparison(config)
    save_comparison_outputs(frame, summary, Path(config.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
