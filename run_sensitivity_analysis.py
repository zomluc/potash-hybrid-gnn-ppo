from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from compare_methods import PPOComparisonSpec, resolve_method_name
from decision_strategy import base_policy, optimize_policy_for_week
from ppo_trainer import load_checkpoint
from rl_environment import PotashSupplyChainEnv
from rl_models import observation_to_torch
from simulation_types import ModelConfig
from supply_chain_dynamics import simulate_horizon
from two_stage_sp import apply_two_stage_plan_one_week, solve_two_stage_sp_for_week


DEFAULT_SENSITIVITY_SEEDS = [611, 722, 833, 944, 1055]


@dataclass(frozen=True)
class SensitivityFactor:
    key: str
    title: str
    levels: tuple[float, ...]
    output_dir: str
    apply_to_config: Callable[[float], ModelConfig]
    level_label: Callable[[float], str]


@dataclass
class SensitivityConfig:
    seeds: list[int]
    device: str = "cpu"
    output_dir: str = "results/sensitivity"
    ppo_specs: list[PPOComparisonSpec] | None = None
    factors: list[SensitivityFactor] | None = None


def default_ppo_specs() -> list[PPOComparisonSpec]:
    return [
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
    ]


def _demand_config(level: float) -> ModelConfig:
    return ModelConfig(average_weekly_demand_tons=float(level))


def _supplier_risk_config(level: float) -> ModelConfig:
    return ModelConfig(supplier_risk_scale=float(level))


def _transport_risk_config(level: float) -> ModelConfig:
    return ModelConfig(transport_risk_scale=float(level))


def _joint_supply_transport_risk_config(level: float) -> ModelConfig:
    return ModelConfig(supplier_risk_scale=float(level), transport_risk_scale=float(level))


def default_factors() -> list[SensitivityFactor]:
    return [
        SensitivityFactor(
            key="demand_scale",
            title="Demand Scale Sensitivity",
            levels=(9000.0, 11000.0, 13000.0, 15000.0),
            output_dir="demand_scale",
            apply_to_config=_demand_config,
            level_label=lambda value: f"{int(value)}",
        ),
        SensitivityFactor(
            key="supply_risk",
            title="Supply Risk Sensitivity",
            levels=(0.5, 1.0, 1.5, 2.0),
            output_dir="supply_risk",
            apply_to_config=_supplier_risk_config,
            level_label=lambda value: f"{value:.2f}x",
        ),
        SensitivityFactor(
            key="transport_risk",
            title="Transport Risk Sensitivity",
            levels=(0.5, 1.0, 1.5, 2.0),
            output_dir="transport_risk",
            apply_to_config=_transport_risk_config,
            level_label=lambda value: f"{value:.2f}x",
        ),
        SensitivityFactor(
            key="joint_supply_transport_risk",
            title="Joint Supply and Transport Risk Stress Test",
            levels=(1.0, 1.5, 2.0, 2.5),
            output_dir="joint_supply_transport_risk",
            apply_to_config=_joint_supply_transport_risk_config,
            level_label=lambda value: f"{value:.2f}x + {value:.2f}x",
        ),
    ]


def default_sensitivity_config() -> SensitivityConfig:
    return SensitivityConfig(
        seeds=list(DEFAULT_SENSITIVITY_SEEDS),
        ppo_specs=default_ppo_specs(),
        factors=default_factors(),
    )


def ensure_checkpoint_exists(spec: PPOComparisonSpec) -> Path:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Retrain all methods before running sensitivity analysis."
        )
    return checkpoint_path


def load_models(ppo_specs: Iterable[PPOComparisonSpec], device: str) -> list[tuple[Any, str, Path]]:
    models: list[tuple[Any, str, Path]] = []
    for spec in ppo_specs:
        checkpoint_path = ensure_checkpoint_exists(spec)
        model, payload, _ = load_checkpoint(checkpoint_path, device=device)
        payload_model_type = payload.get("config", {}).get("model_type", spec.model_type)
        method_name = spec.method_label or resolve_method_name(payload_model_type)
        del payload
        model.eval()
        models.append((model, method_name, checkpoint_path))
    return models


def _empty_episode_stats(method: str, seed: int, factor_key: str, level: float, level_label: str) -> dict[str, Any]:
    return {
        "factor": factor_key,
        "level": float(level),
        "level_label": level_label,
        "method": method,
        "seed": seed,
        "total_profit": 0.0,
        "service_rate": 0.0,
        "weekly_profit_cvar20": 0.0,
    }


def _finalize_episode_stats(stats: dict[str, Any], total_shortage: float, total_demand: float, weekly_profit: list[float]) -> dict[str, Any]:
    stats["service_rate"] = 1.0 - float(total_shortage) / max(float(total_demand), 1.0)
    if weekly_profit:
        ordered = sorted(weekly_profit)
        count = max(1, int(np.ceil(len(ordered) * 0.2)))
        stats["weekly_profit_cvar20"] = float(np.mean(ordered[:count]))
    return stats


def evaluate_baseline_episode(seed: int, config: ModelConfig, factor_key: str, level: float, level_label: str) -> dict[str, Any]:
    env = PotashSupplyChainEnv(config=config)
    observation, _ = env.reset(seed=seed)
    del observation
    stats = _empty_episode_stats("stochastic_planning", seed, factor_key, level, level_label)
    current_policy = base_policy()
    done = False
    weekly_profit: list[float] = []
    total_shortage = 0.0
    total_demand = 0.0

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
        _, _, terminated, truncated, info = env.step(action)
        metrics = info["metrics"]
        stats["total_profit"] += float(metrics["weekly_profit"])
        total_shortage += float(metrics["shortage"])
        total_demand += float(metrics["demand"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        done = terminated or truncated

    return _finalize_episode_stats(stats, total_shortage, total_demand, weekly_profit)


def evaluate_two_stage_sp_episode(seed: int, config: ModelConfig, factor_key: str, level: float, level_label: str) -> dict[str, Any]:
    env = PotashSupplyChainEnv(config=config)
    observation, _ = env.reset(seed=seed)
    del observation
    stats = _empty_episode_stats("two_stage_sp", seed, factor_key, level, level_label)
    done = False
    weekly_profit: list[float] = []
    total_shortage = 0.0
    total_demand = 0.0

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
        env.week += 1
        stats["total_profit"] += float(metrics["weekly_profit"])
        total_shortage += float(metrics["shortage"])
        total_demand += float(metrics["demand"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        done = env.week >= env.config.horizon_weeks

    return _finalize_episode_stats(stats, total_shortage, total_demand, weekly_profit)


def evaluate_ppo_episode(
    model: Any,
    seed: int,
    env_config: ModelConfig,
    device: str,
    method_name: str,
    factor_key: str,
    level: float,
    level_label: str,
) -> dict[str, Any]:
    env = PotashSupplyChainEnv(config=env_config)
    observation, _ = env.reset(seed=seed)
    stats = _empty_episode_stats(method_name, seed, factor_key, level, level_label)
    done = False
    weekly_profit: list[float] = []
    total_shortage = 0.0
    total_demand = 0.0

    while not done:
        obs_t = observation_to_torch(observation, device=device)
        with torch.no_grad():
            action, _, _ = model.act(obs_t, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        metrics = info["metrics"]
        stats["total_profit"] += float(metrics["weekly_profit"])
        total_shortage += float(metrics["shortage"])
        total_demand += float(metrics["demand"])
        weekly_profit.append(float(metrics["weekly_profit"]))
        done = terminated or truncated

    return _finalize_episode_stats(stats, total_shortage, total_demand, weekly_profit)


def summarize_factor_results(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(["factor", "level", "level_label", "method"], as_index=False).agg(
        total_profit_mean=("total_profit", "mean"),
        total_profit_std=("total_profit", "std"),
        service_rate_mean=("service_rate", "mean"),
        service_rate_std=("service_rate", "std"),
        weekly_profit_cvar20_mean=("weekly_profit_cvar20", "mean"),
        weekly_profit_cvar20_std=("weekly_profit_cvar20", "std"),
    )
    return summary


def save_factor_plots(summary: pd.DataFrame, factor: SensitivityFactor, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    methods = sorted(summary["method"].unique())
    level_labels = [factor.level_label(level) for level in factor.levels]
    x = np.arange(len(factor.levels), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    metric_specs = [
        ("total_profit_mean", "Total Profit", "Million USD", 1e6),
        ("service_rate_mean", "Service Rate", "Ratio", 1.0),
        ("weekly_profit_cvar20_mean", "Weekly Profit CVaR20", "Million USD", 1e6),
    ]

    for axis, (metric_name, title, ylabel, scale) in zip(axes, metric_specs):
        for method in methods:
            subset = summary[summary["method"] == method].sort_values("level")
            axis.plot(x, subset[metric_name].to_numpy(dtype=float) / scale, marker="o", linewidth=2, label=method)
        axis.set_title(title)
        axis.set_xlabel("Level")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(level_labels)
        if metric_name == "service_rate_mean":
            axis.set_ylim(0.0, 1.05)
        axis.legend(loc="best")

    fig.suptitle(factor.title)
    fig.tight_layout()
    fig.savefig(output_dir / f"{factor.output_dir}_overview.png", dpi=220)
    plt.close(fig)


def save_factor_outputs(frame: pd.DataFrame, factor: SensitivityFactor, factor_output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    factor_output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_factor_results(frame)
    frame.to_csv(factor_output_dir / "episode_metrics.csv", index=False)
    summary.to_csv(factor_output_dir / "summary.csv", index=False)
    save_factor_plots(summary, factor, factor_output_dir)
    summary_payload = {
        "factor": factor.key,
        "title": factor.title,
        "levels": [float(level) for level in factor.levels],
        "level_labels": [factor.level_label(level) for level in factor.levels],
        "seeds": seeds,
        "summary_by_method": summary.to_dict(orient="records"),
    }
    with (factor_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    return summary_payload


def run_factor_analysis(
    factor: SensitivityFactor,
    models: list[tuple[Any, str, Path]],
    seeds: list[int],
    device: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    print(json.dumps({
        "event": "factor_start",
        "factor": factor.key,
        "title": factor.title,
        "levels": [factor.level_label(level) for level in factor.levels],
    }, ensure_ascii=False))
    for level in factor.levels:
        env_config = factor.apply_to_config(level)
        level_label = factor.level_label(level)
        print(json.dumps({
            "event": "factor_level_start",
            "factor": factor.key,
            "level": float(level),
            "level_label": level_label,
        }, ensure_ascii=False))
        for seed in seeds:
            records.append(evaluate_baseline_episode(seed, env_config, factor.key, level, level_label))
            records.append(evaluate_two_stage_sp_episode(seed, env_config, factor.key, level, level_label))
            for model, method_name, _ in models:
                records.append(
                    evaluate_ppo_episode(
                        model=model,
                        seed=seed,
                        env_config=env_config,
                        device=device,
                        method_name=method_name,
                        factor_key=factor.key,
                        level=level,
                        level_label=level_label,
                    )
                )
        print(json.dumps({
            "event": "factor_level_complete",
            "factor": factor.key,
            "level": float(level),
            "level_label": level_label,
        }, ensure_ascii=False))
    print(json.dumps({
        "event": "factor_complete",
        "factor": factor.key,
        "title": factor.title,
    }, ensure_ascii=False))
    return pd.DataFrame(records)


def run_sensitivity_analysis(config: SensitivityConfig | None = None) -> dict[str, Any]:
    config = config or default_sensitivity_config()
    ppo_specs = config.ppo_specs or []
    factors = config.factors or []
    print(json.dumps({
        "event": "sensitivity_analysis_start",
        "seeds": config.seeds,
        "factors": [factor.key for factor in factors],
    }, ensure_ascii=False))
    models = load_models(ppo_specs, device=config.device)
    root_output_dir = Path(config.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary: dict[str, Any] = {
        "seeds": config.seeds,
        "checkpoint_paths": [
            {"method": method_name, "checkpoint_path": str(checkpoint_path)}
            for _, method_name, checkpoint_path in models
        ],
        "factors": [],
    }

    for factor in factors:
        frame = run_factor_analysis(factor, models=models, seeds=config.seeds, device=config.device)
        factor_output_dir = root_output_dir / factor.output_dir
        summary_payload = save_factor_outputs(frame, factor, factor_output_dir, config.seeds)
        overall_summary["factors"].append(summary_payload)

    for model, _, _ in models:
        del model

    with (root_output_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "event": "sensitivity_analysis_complete",
        "output_dir": str(root_output_dir),
    }, ensure_ascii=False))
    return overall_summary


def main() -> None:
    summary = run_sensitivity_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
