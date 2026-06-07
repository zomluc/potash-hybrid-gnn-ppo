from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_strategy import base_policy, optimize_policy_for_week
from simulation_types import ModelConfig, Policy, Route, Supplier
from supply_chain_environment import build_demand_profile, build_routes, build_suppliers, generate_market_paths, initial_state
from supply_chain_dynamics import simulate_horizon, simulate_one_week


def run_fixed_policy(
    policy: Policy,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    market: dict,
    base_demand: np.ndarray,
) -> pd.DataFrame:
    state = initial_state(config, base_demand)
    records = []
    for week in range(config.horizon_weeks):
        record = simulate_one_week(state, week, policy, config, suppliers, routes, market, base_demand)
        record.update({
            "chosen_la_share": policy.supplier_weights["LA"],
            "chosen_ca_share": policy.supplier_weights["CA"],
            "chosen_ru_share": policy.supplier_weights["RU"],
            "chosen_jo_share": policy.supplier_weights["JO"],
        })
        records.append(record)
    return pd.DataFrame(records)


def run_simulation() -> tuple[pd.DataFrame, dict]:
    config = ModelConfig()
    suppliers = build_suppliers()
    routes = build_routes()
    base_demand = build_demand_profile(config)
    actual_market = generate_market_paths(config, suppliers, routes, seed=20260308)
    state = initial_state(config, base_demand)
    current_policy = base_policy()
    records = []

    for week in range(config.horizon_weeks):
        current_policy = optimize_policy_for_week(
            state,
            week,
            current_policy,
            config,
            suppliers,
            routes,
            actual_market,
            base_demand,
            simulate_horizon,
        )
        record = simulate_one_week(state, week, current_policy, config, suppliers, routes, actual_market, base_demand)
        record.update({
            "chosen_la_share": current_policy.supplier_weights["LA"],
            "chosen_ca_share": current_policy.supplier_weights["CA"],
            "chosen_ru_share": current_policy.supplier_weights["RU"],
            "chosen_jo_share": current_policy.supplier_weights["JO"],
        })
        records.append(record)

    frame = pd.DataFrame(records)
    baseline_frame = run_fixed_policy(base_policy(), config, suppliers, routes, actual_market, base_demand)
    summary = {
        "horizon_weeks": config.horizon_weeks,
        "lookahead_weeks": config.lookahead_weeks,
        "total_profit_usd": float(frame["weekly_profit"].sum()),
        "average_weekly_profit_usd": float(frame["weekly_profit"].mean()),
        "service_rate": float(frame["satisfied"].sum() / frame["demand"].sum()),
        "total_demand_tons": float(frame["demand"].sum()),
        "total_shortage_tons": float(frame["shortage"].sum()),
        "average_inventory_tons": float(frame["inventory"].mean()),
        "ending_inventory_tons": float(frame.iloc[-1]["inventory"]),
        "baseline_profit_usd": float(baseline_frame["weekly_profit"].sum()),
        "baseline_service_rate": float(baseline_frame["satisfied"].sum() / baseline_frame["demand"].sum()),
        "profit_improvement_vs_baseline_usd": float(frame["weekly_profit"].sum() - baseline_frame["weekly_profit"].sum()),
        "recommended_policy": {
            "supplier_weights": {key: round(value, 4) for key, value in current_policy.supplier_weights.items()},
            "safety_stock_weeks": round(current_policy.safety_stock_weeks, 4),
            "reserve_supplier_ratio": round(current_policy.reserve_supplier_ratio, 4),
            "backup_route_ratio": round(current_policy.backup_route_ratio, 4),
            "transport_reserve_ratio": round(current_policy.transport_reserve_ratio, 4),
            "futures_hedge_ratio": round(current_policy.futures_hedge_ratio, 4),
        },
    }
    return frame, summary


def save_outputs(frame: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "weekly_results.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(frame["week"], frame["inventory"], color="#1f77b4")
    axes[0, 0].set_title("Inventory")
    axes[0, 0].set_xlabel("Week")
    axes[0, 0].set_ylabel("Tons")

    axes[0, 1].plot(frame["week"], frame["weekly_profit"], color="#2ca02c")
    axes[0, 1].set_title("Weekly Profit")
    axes[0, 1].set_xlabel("Week")
    axes[0, 1].set_ylabel("USD")

    axes[1, 0].plot(frame["week"], frame["service_rate"], color="#d62728")
    axes[1, 0].set_title("Service Rate")
    axes[1, 0].set_xlabel("Week")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].set_ylim(0.0, 1.05)

    axes[1, 1].stackplot(
        frame["week"],
        frame["chosen_la_share"],
        frame["chosen_ca_share"],
        frame["chosen_ru_share"],
        frame["chosen_jo_share"],
        labels=["Laos", "Canada", "Russia", "Jordan"],
        alpha=0.85,
    )
    axes[1, 1].set_title("Supplier Mix")
    axes[1, 1].set_xlabel("Week")
    axes[1, 1].set_ylabel("Share")
    axes[1, 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "simulation_dashboard.png", dpi=220)
    plt.close(fig)


def main() -> None:
    frame, summary = run_simulation()
    save_outputs(frame, summary, Path("results"))
    print("Imported Potash Supply Chain Dynamic Simulation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
