from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from compare_methods import DEFAULT_COMPARISON_SEEDS, PPOComparisonSpec, default_comparison_config, ensure_checkpoint, resolve_method_name
from decision_strategy import base_policy, optimize_policy_for_week
from ppo_trainer import load_checkpoint
from rl_environment import PotashSupplyChainEnv
from rl_models import observation_to_torch
from simulation_types import Policy
from supply_chain_dynamics import simulate_horizon
from two_stage_sp import apply_two_stage_plan_one_week, route_plan_to_policy, solve_two_stage_sp_for_week


@dataclass
class PolicyAnalysisConfig:
    seeds: list[int]
    output_dir: str = "results/policy_behavior"
    device: str = "cpu"
    ppo_specs: list[PPOComparisonSpec] | None = None


def default_policy_analysis_config() -> PolicyAnalysisConfig:
    comparison_config = default_comparison_config()
    return PolicyAnalysisConfig(
        seeds=list(DEFAULT_COMPARISON_SEEDS),
        ppo_specs=comparison_config.ppo_specs,
    )


def _supplier_entropy(policy: Policy) -> float:
    weights = np.asarray(list(policy.supplier_weights.values()), dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)
    return float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))


def _policy_row(policy: Policy) -> dict[str, float]:
    return {
        "supplier_weight_la": float(policy.supplier_weights["LA"]),
        "supplier_weight_ca": float(policy.supplier_weights["CA"]),
        "supplier_weight_ru": float(policy.supplier_weights["RU"]),
        "supplier_weight_jo": float(policy.supplier_weights["JO"]),
        "supplier_entropy": _supplier_entropy(policy),
        "safety_stock_weeks": float(policy.safety_stock_weeks),
        "reserve_supplier_ratio": float(policy.reserve_supplier_ratio),
        "backup_route_ratio": float(policy.backup_route_ratio),
        "transport_reserve_ratio": float(policy.transport_reserve_ratio),
        "futures_hedge_ratio": float(policy.futures_hedge_ratio),
    }


def _risk_row(risk: dict[str, float]) -> dict[str, float]:
    return {
        "supply_risk": float(risk["current_supply_risk"]),
        "transport_risk": float(risk["current_transport_risk"]),
        "price_risk": float(risk["current_price_risk"]),
    }


def evaluate_baseline_trace(seed: int) -> list[dict[str, Any]]:
    env = PotashSupplyChainEnv()
    observation, _ = env.reset(seed=seed)
    del observation
    method_name = "stochastic_planning"
    current_policy = base_policy()
    done = False
    rows: list[dict[str, Any]] = []

    while not done:
        assert env.state is not None
        assert env.market is not None
        risk_before = env.last_info["risk_summary"]
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
        rows.append(
            {
                "method": method_name,
                "seed": seed,
                "week": int(info["week"]),
                "reward": float(reward),
                "weekly_profit": float(metrics["weekly_profit"]),
                "inventory": float(metrics["inventory"]),
                "backlog": float(metrics["backlog"]),
                "shortage": float(metrics["shortage"]),
                "service_rate": float(metrics["service_rate"]),
                **_risk_row(risk_before),
                **_policy_row(current_policy),
            }
        )
        done = terminated or truncated
    return rows


def evaluate_two_stage_sp_trace(seed: int) -> list[dict[str, Any]]:
    env = PotashSupplyChainEnv()
    observation, _ = env.reset(seed=seed)
    del observation
    method_name = "two_stage_sp"
    done = False
    rows: list[dict[str, Any]] = []

    while not done:
        assert env.state is not None
        assert env.market is not None
        risk_before = env.last_info["risk_summary"]
        plan = solve_two_stage_sp_for_week(
            env.state,
            env.week,
            env.config,
            env.suppliers,
            env.routes,
            env.market,
            env.base_demand,
        )
        policy = route_plan_to_policy(plan.route_quantities, env.state, env.week, env.config, env.routes, env.base_demand)
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
        info = {
            "week": env.week,
            "metrics": metrics,
            "risk_summary": env._risk_info_dict(),
        }
        env.last_info = info
        rows.append(
            {
                "method": method_name,
                "seed": seed,
                "week": int(info["week"]),
                "reward": reward,
                "weekly_profit": float(metrics["weekly_profit"]),
                "inventory": float(metrics["inventory"]),
                "backlog": float(metrics["backlog"]),
                "shortage": float(metrics["shortage"]),
                "service_rate": float(metrics["service_rate"]),
                **_risk_row(risk_before),
                **_policy_row(policy),
            }
        )
        done = env.week >= env.config.horizon_weeks
    return rows


def evaluate_ppo_trace(model: Any, seed: int, method_name: str, device: str) -> list[dict[str, Any]]:
    env = PotashSupplyChainEnv()
    observation, _ = env.reset(seed=seed)
    done = False
    rows: list[dict[str, Any]] = []

    while not done:
        risk_before = env.last_info["risk_summary"]
        obs_t = observation_to_torch(observation, device=device)
        with torch.no_grad():
            action, _, _ = model.act(obs_t, deterministic=True)
        action_np = action.detach().cpu().numpy()
        policy = env.decode_action(action_np)
        observation, reward, terminated, truncated, info = env.step(action_np)
        metrics = info["metrics"]
        rows.append(
            {
                "method": method_name,
                "seed": seed,
                "week": int(info["week"]),
                "reward": float(reward),
                "weekly_profit": float(metrics["weekly_profit"]),
                "inventory": float(metrics["inventory"]),
                "backlog": float(metrics["backlog"]),
                "shortage": float(metrics["shortage"]),
                "service_rate": float(metrics["service_rate"]),
                **_risk_row(risk_before),
                **_policy_row(policy),
            }
        )
        done = terminated or truncated
    return rows


def _risk_regime_summary(frame: pd.DataFrame, risk_column: str, control_columns: list[str]) -> pd.DataFrame:
    threshold = float(frame[risk_column].quantile(0.75))
    labeled = frame.copy()
    labeled["risk_regime"] = np.where(labeled[risk_column] >= threshold, f"high_{risk_column}", "normal")
    return labeled.groupby(["method", "risk_regime"], as_index=False).agg(
        weekly_profit_mean=("weekly_profit", "mean"),
        service_rate_mean=("service_rate", "mean"),
        shortage_mean=("shortage", "mean"),
        **{f"{column}_mean": (column, "mean") for column in control_columns},
    )


def summarize_policy_behavior(frame: pd.DataFrame) -> dict[str, Any]:
    control_columns = [
        "supplier_entropy",
        "safety_stock_weeks",
        "reserve_supplier_ratio",
        "backup_route_ratio",
        "transport_reserve_ratio",
        "futures_hedge_ratio",
    ]
    overall = frame.groupby("method", as_index=False).agg(
        weekly_profit_mean=("weekly_profit", "mean"),
        weekly_profit_std=("weekly_profit", "std"),
        service_rate_mean=("service_rate", "mean"),
        shortage_mean=("shortage", "mean"),
        **{f"{column}_mean": (column, "mean") for column in control_columns},
    )
    supply_summary = _risk_regime_summary(frame, "supply_risk", control_columns)
    transport_summary = _risk_regime_summary(frame, "transport_risk", control_columns)
    price_summary = _risk_regime_summary(frame, "price_risk", control_columns)
    return {
        "seeds": sorted(int(seed) for seed in frame["seed"].unique()),
        "overall_by_method": overall.to_dict(orient="records"),
        "supply_risk_regime_summary": supply_summary.to_dict(orient="records"),
        "transport_risk_regime_summary": transport_summary.to_dict(orient="records"),
        "price_risk_regime_summary": price_summary.to_dict(orient="records"),
    }


def save_policy_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    methods = sorted(frame["method"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    metric_specs = [
        ("safety_stock_weeks", "Safety Stock"),
        ("backup_route_ratio", "Backup Route Ratio"),
        ("futures_hedge_ratio", "Futures Hedge Ratio"),
    ]
    for axis, (column, title) in zip(axes, metric_specs):
        for method in methods:
            subset = frame[frame["method"] == method].groupby("week", as_index=False)[column].mean()
            axis.plot(subset["week"], subset[column], linewidth=2, label=method)
        axis.set_title(title)
        axis.set_xlabel("Week")
        axis.set_ylabel(column.replace("_", " ").title())
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "policy_control_trajectories.png", dpi=240)
    plt.close(fig)


def run_policy_behavior_analysis(config: PolicyAnalysisConfig | None = None) -> dict[str, Any]:
    config = config or default_policy_analysis_config()
    ppo_specs = config.ppo_specs or []
    models: list[tuple[Any, str, Path]] = []
    for spec in ppo_specs:
        checkpoint_path = ensure_checkpoint(default_comparison_config(), spec)
        model, payload, _ = load_checkpoint(checkpoint_path, device=config.device)
        payload_model_type = payload.get("config", {}).get("model_type", spec.model_type)
        method_name = spec.method_label or resolve_method_name(payload_model_type)
        del payload
        model.eval()
        models.append((model, method_name, checkpoint_path))

    records: list[dict[str, Any]] = []
    for seed in config.seeds:
        records.extend(evaluate_baseline_trace(seed))
        records.extend(evaluate_two_stage_sp_trace(seed))
        for model, method_name, _ in models:
            records.extend(evaluate_ppo_trace(model, seed, method_name, config.device))

    frame = pd.DataFrame(records)
    summary = summarize_policy_behavior(frame)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "policy_weekly_trace.csv", index=False)
    with (output_dir / "policy_behavior_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    save_policy_plots(frame, output_dir)

    for model, _, _ in models:
        del model
    return summary


def main() -> None:
    summary = run_policy_behavior_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
