from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation_types import ModelConfig
from supply_chain_environment import build_routes, build_suppliers, generate_market_paths


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def build_summary(num_paths: int, seed_start: int) -> dict:
    config = ModelConfig()
    suppliers = build_suppliers(config)
    routes = build_routes()

    annual_demand = []
    weekly_demand_mean = []
    weekly_demand_cv = []
    pooled_weekly_demand = []
    spot_price_mean = {code: [] for code in suppliers}
    spot_price_std = {code: [] for code in suppliers}
    supply_disruption_share = {code: [] for code in suppliers}
    node_disruption_share = {}
    route_delay_share = {name: [] for name in routes}
    positive_route_delay = {name: [] for name in routes}

    for offset in range(num_paths):
        market = generate_market_paths(
            config,
            suppliers,
            routes,
            seed=seed_start + offset,
        )
        demand = np.asarray(market["demand"], dtype=float)
        annual_demand.append(float(np.sum(demand)))
        weekly_demand_mean.append(float(np.mean(demand)))
        weekly_demand_cv.append(float(np.std(demand, ddof=1) / np.mean(demand)))
        pooled_weekly_demand.extend(demand.tolist())

        for code in suppliers:
            price = np.asarray(market["spot_prices"][code], dtype=float)
            spot_price_mean[code].append(float(np.mean(price)))
            spot_price_std[code].append(float(np.std(price, ddof=1)))
            multiplier = np.asarray(market["supply_multipliers"][code], dtype=float)
            supply_disruption_share[code].append(float(np.mean(multiplier < 1.0)))

        for node, multiplier_values in market["node_multipliers"].items():
            node_disruption_share.setdefault(node, []).append(
                float(np.mean(np.asarray(multiplier_values, dtype=float) < 1.0))
            )

        for name in routes:
            delays = np.asarray(market["route_delays"][name], dtype=float)
            route_delay_share[name].append(float(np.mean(delays > 0.0)))
            positive = delays[delays > 0.0]
            if positive.size:
                positive_route_delay[name].extend(positive.tolist())

    return {
        "paths": num_paths,
        "seed_start": seed_start,
        "horizon_weeks": config.horizon_weeks,
        "baseline": {
            "average_weekly_demand_tons": config.average_weekly_demand_tons,
            "supplier_annual_capacity_tons": {
                code: supplier.annual_capacity_tons for code, supplier in suppliers.items()
            },
            "supplier_base_fob_usd_per_ton": {
                code: supplier.base_fob for code, supplier in suppliers.items()
            },
            "supplier_base_risk": {
                code: supplier.base_risk for code, supplier in suppliers.items()
            },
            "route_lead_weeks": {name: route.lead_weeks for name, route in routes.items()},
            "route_weekly_capacity_tons": {
                name: route.base_capacity_tons for name, route in routes.items()
            },
            "route_base_risk": {name: route.base_risk for name, route in routes.items()},
        },
        "generated_moments": {
            "annual_demand_tons": _summary(annual_demand),
            "weekly_demand_mean_tons": _summary(weekly_demand_mean),
            "pooled_weekly_demand_tons": _summary(pooled_weekly_demand),
            "weekly_demand_cv": _summary(weekly_demand_cv),
            "spot_price_annual_mean_usd_per_ton": {
                code: _summary(values) for code, values in spot_price_mean.items()
            },
            "spot_price_within_year_std_usd_per_ton": {
                code: _summary(values) for code, values in spot_price_std.items()
            },
            "supplier_disrupted_week_share": {
                code: _summary(values) for code, values in supply_disruption_share.items()
            },
            "gateway_disrupted_week_share": {
                node: _summary(values) for node, values in node_disruption_share.items()
            },
            "route_delayed_week_share": {
                name: _summary(values) for name, values in route_delay_share.items()
            },
            "route_positive_delay_weeks": {
                name: _summary(values) if values else None
                for name, values in positive_route_delay.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize generated moments used to check environment calibration."
    )
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/calibration/environment_calibration_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.paths, args.seed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
