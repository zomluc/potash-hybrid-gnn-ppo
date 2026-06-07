from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from scenario_tree import build_scenario_tree
from simulation_types import ModelConfig, Policy, Route, State, Supplier


def base_policy() -> Policy:
    return Policy(
        supplier_weights={"LA": 0.48, "CA": 0.20, "RU": 0.18, "JO": 0.14},
        safety_stock_weeks=3.0,
        reserve_supplier_ratio=0.10,
        backup_route_ratio=0.10,
        transport_reserve_ratio=0.08,
        futures_hedge_ratio=0.30,
    ).normalized()


def basket_price(weights: Dict[str, float], values: Dict[str, float]) -> float:
    return sum(weights[key] * values[key] for key in weights)

def choose_route_supplier_split(policy: Policy, market: dict, week: int) -> Dict[str, Dict[str, float]]:
    zj_risk = market["node_prob"]["ZJ"][week]
    weights = dict(policy.supplier_weights)
    if market["supply_multipliers"]["LA"][week] < 0.7:
        weights["LA"] *= 0.7
        weights["CA"] *= 1.1
        weights["RU"] *= 1.1
        weights["JO"] *= 1.05
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}

    backup_ratio = policy.backup_route_ratio
    if zj_risk > 0.18 or market["node_multipliers"]["ZJ"][week] < 0.7:
        backup_ratio = min(0.65, backup_ratio + 0.20)

    route_split = {
        "LA": {"LA_MH": 1.0},
        "CA": {"CA_ZJ": 1.0 - backup_ratio, "CA_FCG": backup_ratio},
        "RU": {"RU_ZJ": 1.0 - backup_ratio, "RU_MZL": backup_ratio},
        "JO": {"JO_LYG": 1.0 - 0.5 * backup_ratio, "JO_ZJ": 0.5 * backup_ratio},
    }
    return {supplier: {route: share * weights[supplier] for route, share in split.items()} for supplier, split in route_split.items()}


def market_risk_snapshot(market: dict, week: int) -> dict[str, float]:
    supply_risk = max(float(market["supply_prob"][code][week]) for code in ["LA", "CA", "RU", "JO"])
    transport_risk = max(float(market["node_prob"][node][week]) for node in ["ZJ", "FCG", "MH", "LYG", "MZL"])
    spot_prices = np.asarray([float(market["spot_prices"][code][week]) for code in ["LA", "CA", "RU", "JO"]], dtype=np.float32)
    price_risk = float(np.std(spot_prices))
    return {
        "supply_risk": supply_risk,
        "transport_risk": transport_risk,
        "price_risk": price_risk,
    }


def generate_candidate_policies(policy_seed: Policy, market: dict, week: int) -> List[Policy]:
    risk = market_risk_snapshot(market, week)
    supplier_sets = []
    base = dict(policy_seed.supplier_weights)
    supplier_sets.append(base)
    supplier_sets.append({"LA": max(0.30, base["LA"] - 0.10), "CA": base["CA"] + 0.06, "RU": base["RU"] + 0.02, "JO": base["JO"] + 0.02})
    supplier_sets.append({"LA": min(0.62, base["LA"] + 0.08), "CA": max(0.12, base["CA"] - 0.03), "RU": base["RU"] - 0.02, "JO": base["JO"] - 0.03})

    if risk["supply_risk"] > 0.08:
        supplier_sets.append({"LA": 0.30, "CA": 0.28, "RU": 0.24, "JO": 0.18})
    if risk["transport_risk"] > 0.16:
        supplier_sets.append({"LA": 0.45, "CA": 0.18, "RU": 0.22, "JO": 0.15})

    safety_choices = [2.5, 3.5, 4.5] if risk["transport_risk"] > 0.12 or risk["supply_risk"] > 0.07 else [2.0, 3.0, 4.0]
    backup_choices = [0.10, 0.30, 0.50] if risk["transport_risk"] > 0.12 else [0.0, 0.15, 0.30]
    reserve_choices = [0.05, 0.15, 0.25]
    futures_choices = [0.15, 0.35, 0.60] if risk["price_risk"] > 9.0 else [0.0, 0.20, 0.40]
    candidates = []
    for supplier_weights in supplier_sets[:4]:
        for safety_stock_weeks in safety_choices[:3]:
            for backup_route_ratio in backup_choices[:2]:
                for futures_hedge_ratio in futures_choices[:2]:
                    transport_reserve_ratio = reserve_choices[1 if backup_route_ratio > 0.0 else 0]
                    candidates.append(
                        Policy(
                            supplier_weights=supplier_weights,
                            safety_stock_weeks=safety_stock_weeks,
                            reserve_supplier_ratio=reserve_choices[1],
                            backup_route_ratio=backup_route_ratio,
                            transport_reserve_ratio=transport_reserve_ratio,
                            futures_hedge_ratio=futures_hedge_ratio,
                        ).normalized()
                    )

    unique = []
    signatures = set()
    for candidate in candidates:
        signature = (
            tuple(round(candidate.supplier_weights[key], 3) for key in sorted(candidate.supplier_weights)),
            round(candidate.safety_stock_weeks, 1),
            round(candidate.backup_route_ratio, 2),
            round(candidate.transport_reserve_ratio, 2),
            round(candidate.futures_hedge_ratio, 2),
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(candidate)
    return unique[:28]


def optimize_policy_for_week(
    state: State,
    week: int,
    current_policy: Policy,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    actual_market: dict,
    base_demand: np.ndarray,
    horizon_simulator,
) -> Policy:
    candidates = generate_candidate_policies(current_policy, actual_market, week)
    current_prices = {code: float(actual_market["spot_prices"][code][week]) for code in suppliers}
    best_score = -1e18
    best_policy = current_policy
    forecast = np.concatenate([
        base_demand[week: week + config.lookahead_weeks],
        np.repeat(base_demand[-1], max(0, config.lookahead_weeks - len(base_demand[week: week + config.lookahead_weeks]))),
    ])

    for candidate_index, candidate in enumerate(candidates):
        tree = build_scenario_tree(config, suppliers, routes, week, candidate_index, current_prices)
        profits = []
        for path in tree.paths:
            result = horizon_simulator(
                start_state=state,
                start_week=0,
                policy=candidate,
                config=config,
                suppliers=suppliers,
                routes=routes,
                market=path.market,
                base_demand=forecast,
                horizon=min(config.lookahead_weeks, len(forecast)),
            )
            profits.append(result["profit"])

        # Baseline planning now optimizes the same objective as PPO: total revenue minus total cost.
        expected_profit = float(np.mean(profits))
        score = expected_profit

        if score > best_score:
            best_score = score
            best_policy = candidate

    return best_policy
