from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from simulation_types import ModelConfig, Route, Supplier
from supply_chain_environment import generate_market_paths


@dataclass
class ScenarioPath:
    scenario_id: int
    probability: float
    market: dict


@dataclass
class ScenarioTree:
    horizon_weeks: int
    paths: List[ScenarioPath]


def build_scenario_tree(
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    week: int,
    candidate_index: int,
    start_prices: Dict[str, float],
) -> ScenarioTree:
    paths = []
    probability = 1.0 / config.planning_scenarios
    for scenario_id in range(config.planning_scenarios):
        market = generate_market_paths(
            config,
            suppliers,
            routes,
            seed=10_000 + week * 500 + candidate_index * 50 + scenario_id,
            horizon=config.lookahead_weeks,
            start_prices=start_prices,
        )
        paths.append(ScenarioPath(scenario_id=scenario_id, probability=probability, market=market))
    return ScenarioTree(horizon_weeks=config.lookahead_weeks, paths=paths)


def cvar(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    count = max(1, int(len(ordered) * alpha + 0.999999))
    return float(sum(ordered[:count]) / count)