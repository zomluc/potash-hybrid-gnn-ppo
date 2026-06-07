from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from supply_chain_dynamics import simulate_one_week
from simulation_types import ModelConfig, Policy, State
from supply_chain_environment import PRIMARY_ROUTE, build_demand_profile, build_routes, build_suppliers, generate_market_paths, initial_state


NODE_ORDER = [
    "supplier_LA",
    "supplier_CA",
    "supplier_RU",
    "supplier_JO",
    "port_ZJ",
    "port_FCG",
    "port_MH",
    "port_LYG",
    "port_MZL",
    "focal_firm",
]

ROUTE_ORDER = [
    "LA_MH",
    "CA_ZJ",
    "CA_FCG",
    "RU_ZJ",
    "RU_MZL",
    "JO_LYG",
    "JO_ZJ",
]

PORT_TO_FIRM_EDGES = {
    "ZJ_FIRM": "ZJ",
    "FCG_FIRM": "FCG",
    "MH_FIRM": "MH",
    "LYG_FIRM": "LYG",
    "MZL_FIRM": "MZL",
}

EDGE_ORDER = ROUTE_ORDER + list(PORT_TO_FIRM_EDGES)

EDGE_INDEX = np.array(
    [
        [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
        [6, 4, 5, 4, 8, 7, 4, 9, 9, 9, 9, 9],
    ],
    dtype=np.int32,
)


class PotashSupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: ModelConfig | None = None,
        reward_scale: float = 100_000.0,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.reward_scale = reward_scale
        self.suppliers = build_suppliers(self.config)
        self.routes = build_routes()
        self.base_demand = build_demand_profile(self.config)

        self.node_feature_dim = 10
        self.edge_feature_dim = 9
        self.global_feature_dim = 19
        self.flat_feature_dim = 25
        self.global_history_steps = int(max(1, self.config.gnn_history_steps))
        self.action_dim = 9

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(len(NODE_ORDER), self.node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_features": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(len(EDGE_ORDER), self.edge_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=len(NODE_ORDER) - 1,
                    shape=EDGE_INDEX.shape,
                    dtype=np.int32,
                ),
                "global_features": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
                "global_feature_history": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(self.global_history_steps, self.global_feature_dim),
                    dtype=np.float32,
                ),
                "flat_observation": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(self.flat_feature_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.market: dict[str, Any] | None = None
        self.state: State | None = None
        self.week: int = 0
        self.episode_seed: int = 0
        self.last_info: dict[str, Any] = {}
        self.last_policy: Policy = self.default_policy()
        self.global_feature_history: deque[np.ndarray] = deque(maxlen=self.global_history_steps)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.episode_seed = int(options.get("episode_seed", self.np_random.integers(1, 1_000_000_000)))
        self.base_demand = build_demand_profile(self.config)
        self.market = generate_market_paths(self.config, self.suppliers, self.routes, seed=self.episode_seed)
        self.state = initial_state(self.config, self.base_demand)
        self.week = 0
        self.last_policy = self.default_policy()
        self.global_feature_history = deque(maxlen=self.global_history_steps)
        self.last_info = {
            "episode_seed": self.episode_seed,
            "policy": asdict(self.last_policy),
            "risk_summary": self._risk_info_dict(),
        }
        return self._get_observation(), self.last_info

    def step(self, action: np.ndarray) -> Tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.market is None or self.state is None:
            raise RuntimeError("Environment must be reset before step().")
        if self.week >= self.config.horizon_weeks:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        policy = self.decode_action(action)
        metrics = simulate_one_week(
            self.state,
            self.week,
            policy,
            self.config,
            self.suppliers,
            self.routes,
            self.market,
            self.base_demand,
        )
        reward = float(metrics["weekly_profit"] / self.reward_scale)
        self.last_policy = policy
        self.week += 1
        terminated = self.week >= self.config.horizon_weeks
        truncated = False

        info = {
            "week": self.week,
            "policy": asdict(policy),
            "metrics": metrics,
            "reward_components": {"profit_term": reward},
            "risk_summary": self._risk_info_dict(),
        }
        self.last_info = info
        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.state is None:
            print("Environment not initialized.")
            return
        print(
            {
                "week": self.week,
                "inventory": round(self.state.inventory, 2),
                "backlog": round(self.state.backlog, 2),
                "cumulative_profit": round(self.state.cumulative_profit, 2),
            }
        )

    def default_policy(self) -> Policy:
        return Policy(
            supplier_weights={"LA": 0.48, "CA": 0.20, "RU": 0.18, "JO": 0.14},
            safety_stock_weeks=3.0,
            reserve_supplier_ratio=0.10,
            backup_route_ratio=0.10,
            transport_reserve_ratio=0.08,
            futures_hedge_ratio=0.30,
        ).normalized()

    def decode_action(self, action: np.ndarray) -> Policy:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {action.shape[0]}")

        logits = action[:4]
        logits = logits - np.max(logits)
        weights = np.exp(logits) / np.sum(np.exp(logits))

        def sigmoid(value: float) -> float:
            return 1.0 / (1.0 + np.exp(-float(value)))

        return Policy(
            supplier_weights={"LA": float(weights[0]), "CA": float(weights[1]), "RU": float(weights[2]), "JO": float(weights[3])},
            safety_stock_weeks=1.0 + 5.0 * sigmoid(action[4]),
            reserve_supplier_ratio=0.50 * sigmoid(action[5]),
            backup_route_ratio=0.80 * sigmoid(action[6]),
            transport_reserve_ratio=0.50 * sigmoid(action[7]),
            futures_hedge_ratio=0.95 * sigmoid(action[8]),
        ).normalized()

    def encode_policy(self, policy: Policy) -> np.ndarray:
        policy = policy.normalized()
        weights = np.array([
            policy.supplier_weights["LA"],
            policy.supplier_weights["CA"],
            policy.supplier_weights["RU"],
            policy.supplier_weights["JO"],
        ])
        logits = np.log(np.maximum(weights, 1e-8))

        def inverse_sigmoid(value: float, low: float, high: float) -> float:
            scaled = np.clip((value - low) / (high - low), 1e-6, 1.0 - 1e-6)
            return float(np.log(scaled / (1.0 - scaled)))

        return np.array(
            [
                logits[0],
                logits[1],
                logits[2],
                logits[3],
                inverse_sigmoid(policy.safety_stock_weeks, 1.0, 6.0),
                inverse_sigmoid(policy.reserve_supplier_ratio, 0.0, 0.50),
                inverse_sigmoid(policy.backup_route_ratio, 0.0, 0.80),
                inverse_sigmoid(policy.transport_reserve_ratio, 0.0, 0.50),
                inverse_sigmoid(policy.futures_hedge_ratio, 0.0, 0.95),
            ],
            dtype=np.float32,
        )

    def sample_heuristic_action(self) -> np.ndarray:
        return self.encode_policy(self.default_policy())

    def _pipeline_buckets(self) -> tuple[float, float, float]:
        assert self.state is not None
        inbound_1w = 0.0
        inbound_2_3w = 0.0
        inbound_4pw = 0.0
        for item in self.state.shipments:
            delta = item.arrival_week - self.week
            if delta <= 0:
                continue
            if delta <= 1:
                inbound_1w += item.quantity
            elif delta <= 3:
                inbound_2_3w += item.quantity
            else:
                inbound_4pw += item.quantity
        return float(inbound_1w), float(inbound_2_3w), float(inbound_4pw)

    def _pipeline_metadata(self) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        route_pipeline_qty = {name: 0.0 for name in ROUTE_ORDER}
        route_eta_weighted = {name: 0.0 for name in ROUTE_ORDER}
        supplier_pipeline_qty = {code: 0.0 for code in self.suppliers}
        node_pipeline_qty = {node: 0.0 for node in ["ZJ", "FCG", "MH", "LYG", "MZL"]}

        assert self.state is not None
        for item in self.state.shipments:
            delta = item.arrival_week - self.week
            if delta <= 0:
                continue
            quantity = float(item.quantity)
            route_name = item.route_name
            supplier_code = item.supplier_code
            if route_name in route_pipeline_qty:
                route_pipeline_qty[route_name] += quantity
                route_eta_weighted[route_name] += quantity * float(delta)
                node_pipeline_qty[self.routes[route_name].node] += quantity
            if supplier_code in supplier_pipeline_qty:
                supplier_pipeline_qty[supplier_code] += quantity

        route_eta_avg = {
            name: (route_eta_weighted[name] / route_pipeline_qty[name] if route_pipeline_qty[name] > 1e-6 else 0.0)
            for name in ROUTE_ORDER
        }
        return route_pipeline_qty, route_eta_avg, supplier_pipeline_qty, node_pipeline_qty

    def _route_exposure_from_policy(self, policy: Policy) -> dict[str, float]:
        route_exposure = {name: 0.0 for name in ROUTE_ORDER}
        for supplier_code, supplier_weight in policy.supplier_weights.items():
            primary_route_name = PRIMARY_ROUTE[supplier_code]
            primary_route = self.routes[primary_route_name]
            backup_route_name = primary_route.backup_route
            if backup_route_name is None:
                route_exposure[primary_route_name] += supplier_weight
                continue
            backup_share = supplier_weight * policy.backup_route_ratio
            primary_share = supplier_weight - backup_share
            route_exposure[primary_route_name] += primary_share
            route_exposure[backup_route_name] += backup_share
        return route_exposure

    def _node_exposure_from_policy(self, route_exposure: dict[str, float]) -> dict[str, float]:
        node_exposure = {node: 0.0 for node in ["ZJ", "FCG", "MH", "LYG", "MZL"]}
        for route_name, exposure in route_exposure.items():
            node_exposure[self.routes[route_name].node] += exposure
        return node_exposure

    def _policy_summary(self, policy: Policy) -> dict[str, float]:
        weights = np.array(list(policy.supplier_weights.values()), dtype=np.float32)
        return {
            "supplier_concentration": float(np.sum(np.square(weights))),
            "max_supplier_weight": float(np.max(weights)),
            "safety_stock_weeks": float(policy.safety_stock_weeks),
            "reserve_supplier_ratio": float(policy.reserve_supplier_ratio),
            "backup_route_ratio": float(policy.backup_route_ratio),
            "transport_reserve_ratio": float(policy.transport_reserve_ratio),
            "futures_hedge_ratio": float(policy.futures_hedge_ratio),
        }

    def _demand_normalization_scale(self) -> float:
        peak_multiplier = max(1.0, float(self.config.demand_peak_seasonality))
        return float(max(15_000.0, self.config.average_weekly_demand_tons * peak_multiplier))

    def _compact_market_summary(self, current_week: int) -> dict[str, float]:
        supplier_probs = [float(self.market["supply_prob"][code][current_week]) for code in self.suppliers]
        supplier_multipliers = [float(self.market["supply_multipliers"][code][current_week]) for code in self.suppliers]
        node_probs = [float(self.market["node_prob"][node][current_week]) for node in ["ZJ", "FCG", "MH", "LYG", "MZL"]]
        route_probs = [float(self.market["route_prob"][route][current_week]) for route in ROUTE_ORDER]
        node_multipliers = [float(self.market["node_multipliers"][node][current_week]) for node in ["ZJ", "FCG", "MH", "LYG", "MZL"]]
        return {
            "max_supply_prob": max(supplier_probs),
            "min_supply_multiplier": min(supplier_multipliers),
            "max_node_prob": max(node_probs + route_probs),
            "min_node_multiplier": min(node_multipliers),
        }

    def _exogenous_signal_summary(self, current_week: int) -> dict[str, float]:
        if self.market is None:
            raise RuntimeError("Environment state is not initialized.")

        signals = self.market["observable_signals"]
        seasonal = self.market["seasonal_profiles"]

        supply_warning = max(
            float(signals["mine_accident"][current_week]),
            float(signals["sanctions_tightening"][current_week]),
            float(signals["maintenance_warning"][current_week]),
        )
        transport_warning = max(
            float(signals["port_congestion_alert"][current_week]),
            float(signals["storm_alert"][current_week]),
            float(signals["rail_disruption_alert"][current_week]),
        )
        price_warning = max(
            float(signals["energy_shock"][current_week]),
            float(signals["demand_surge_signal"][current_week]),
            float(signals["mine_accident"][current_week]),
            float(signals["sanctions_tightening"][current_week]),
        )

        return {
            "planting_peak": float(seasonal["planting_peak"][current_week]),
            "monsoon_pressure": float(seasonal["monsoon_pressure"][current_week]),
            "maintenance_cycle": float(seasonal["maintenance_cycle"][current_week]),
            "winter_energy": float(seasonal["winter_energy"][current_week]),
            "supply_warning": supply_warning,
            "transport_warning": transport_warning,
            "price_warning": price_warning,
        }

    def _risk_info_dict(self) -> dict[str, float]:
        if self.market is None:
            return {
                "current_supply_risk": 0.0,
                "current_transport_risk": 0.0,
                "current_price_risk": 0.0,
            }
        current_week = min(self.week, self.config.horizon_weeks - 1)
        market_summary = self._compact_market_summary(current_week)
        spot_prices = [float(self.market["spot_prices"][code][current_week]) for code in ["LA", "CA", "RU", "JO"]]
        price_risk = float(np.clip(np.std(spot_prices) / 50.0, 0.0, 1.0))
        return {
            "current_supply_risk": float(market_summary["max_supply_prob"]),
            "current_transport_risk": float(market_summary["max_node_prob"]),
            "current_price_risk": price_risk,
        }

    def _update_global_history(self, global_features: np.ndarray) -> np.ndarray:
        if not self.global_feature_history or not np.array_equal(self.global_feature_history[-1], global_features):
            self.global_feature_history.append(global_features.copy())
        history = np.zeros((self.global_history_steps, self.global_feature_dim), dtype=np.float32)
        recent = list(self.global_feature_history)
        history[-len(recent):] = np.asarray(recent, dtype=np.float32)
        return history

    def _get_observation(self) -> dict[str, np.ndarray]:
        if self.market is None or self.state is None:
            raise RuntimeError("Environment state is not initialized.")

        current_week = min(self.week, self.config.horizon_weeks - 1)
        current_demand = float(self.market["demand"][current_week])
        backlog = float(self.state.backlog)
        pipeline_qty = float(sum(item.quantity for item in self.state.shipments if item.arrival_week > self.week))
        inbound_1w, inbound_2_3w, inbound_4pw = self._pipeline_buckets()
        basket_spot = float(np.mean([self.market["spot_prices"][code][current_week] for code in ["LA", "CA", "RU", "JO"]]))
        route_exposure = self._route_exposure_from_policy(self.last_policy)
        node_exposure = self._node_exposure_from_policy(route_exposure)
        route_pipeline_qty, route_eta_avg, supplier_pipeline_qty, node_pipeline_qty = self._pipeline_metadata()
        policy_summary = self._policy_summary(self.last_policy)
        demand_scale = self._demand_normalization_scale()
        market_summary = self._compact_market_summary(current_week)
        exogenous_summary = self._exogenous_signal_summary(current_week)
        remaining_weeks = float(max(self.config.horizon_weeks - self.week, 0))
        remaining_weeks_frac = remaining_weeks / max(float(self.config.horizon_weeks), 1.0)
        week_phase = 2.0 * np.pi * float(current_week) / max(float(self.config.horizon_weeks), 1.0)
        week_sin = float(np.sin(week_phase))
        week_cos = float(np.cos(week_phase))
        max_route_pipeline_ratio = max(route_pipeline_qty.values(), default=0.0) / max(pipeline_qty, 1.0)
        max_supplier_pipeline_ratio = max(supplier_pipeline_qty.values(), default=0.0) / max(pipeline_qty, 1.0)

        node_features = np.zeros((len(NODE_ORDER), self.node_feature_dim), dtype=np.float32)
        for idx, node_name in enumerate(NODE_ORDER):
            feature = np.zeros(self.node_feature_dim, dtype=np.float32)
            if node_name.startswith("supplier"):
                code = node_name.split("_")[-1]
                feature[:3] = [1.0, 0.0, 0.0]
                feature[3] = np.clip(self.suppliers[code].weekly_capacity / 50_000.0, 0.0, 5.0)
                feature[4] = np.clip(float(self.market["spot_prices"][code][current_week]) / 500.0, 0.0, 2.0)
                feature[5] = float(self.market["supply_prob"][code][current_week])
                feature[6] = float(self.market["supply_multipliers"][code][current_week])
                feature[7] = float(self.last_policy.supplier_weights[code])
                feature[8] = np.clip(supplier_pipeline_qty[code] / 50_000.0, 0.0, 5.0)
                feature[9] = remaining_weeks_frac
            elif node_name.startswith("port"):
                code = node_name.split("_")[-1]
                incoming_capacity = sum(route.base_capacity_tons for route in self.routes.values() if route.node == code)
                route_costs = [route.base_freight + route.handling_cost for route in self.routes.values() if route.node == code]
                avg_route_cost = float(np.mean(route_costs)) if route_costs else 0.0
                feature[:3] = [0.0, 1.0, 0.0]
                feature[3] = np.clip(incoming_capacity / 20_000.0, 0.0, 5.0)
                feature[4] = np.clip(avg_route_cost / 120.0, 0.0, 3.0)
                feature[5] = float(self.market["node_prob"][code][current_week])
                feature[6] = float(self.market["node_multipliers"][code][current_week])
                feature[7] = float(node_exposure[code])
                feature[8] = np.clip(node_pipeline_qty[code] / 30_000.0, 0.0, 5.0)
                feature[9] = week_sin
            else:
                feature[:3] = [0.0, 0.0, 1.0]
                feature[3] = np.clip(self.state.inventory / 50_000.0, 0.0, 5.0)
                feature[4] = np.clip(pipeline_qty / 50_000.0, 0.0, 5.0)
                feature[5] = np.clip(backlog / 10_000.0, 0.0, 5.0)
                feature[6] = np.clip(inbound_1w / 20_000.0, 0.0, 5.0)
                feature[7] = np.clip(current_demand / demand_scale, 0.0, 5.0)
                feature[8] = np.clip(inbound_4pw / 40_000.0, 0.0, 5.0)
                feature[9] = remaining_weeks_frac
            node_features[idx] = feature

        edge_features = np.zeros((len(EDGE_ORDER), self.edge_feature_dim), dtype=np.float32)
        for idx, route_name in enumerate(ROUTE_ORDER):
            route = self.routes[route_name]
            feature = np.zeros(self.edge_feature_dim, dtype=np.float32)
            if route.mode == "sea":
                feature[:3] = [1.0, 0.0, 0.0]
            elif route.mode == "rail":
                feature[:3] = [0.0, 1.0, 0.0]
            else:
                feature[:3] = [0.0, 0.0, 1.0]
            feature[3] = route.lead_weeks / 6.0
            feature[4] = np.clip((route.base_freight + route.handling_cost) / 120.0, 0.0, 3.0)
            feature[5] = max(
                float(self.market["supply_prob"][route.supplier][current_week]),
                float(self.market["node_prob"][route.node][current_week]),
                float(self.market["route_prob"][route_name][current_week]),
            )
            feature[6] = float(route_exposure[route_name])
            feature[7] = np.clip(route_pipeline_qty[route_name] / 20_000.0, 0.0, 5.0)
            feature[8] = np.clip(route_eta_avg[route_name] / 6.0, 0.0, 3.0)
            edge_features[idx] = feature
        for offset, (edge_name, node) in enumerate(PORT_TO_FIRM_EDGES.items(), start=len(ROUTE_ORDER)):
            feature = np.zeros(self.edge_feature_dim, dtype=np.float32)
            feature[5] = float(self.market["node_prob"][node][current_week])
            feature[6] = float(node_exposure[node])
            feature[7] = np.clip(node_pipeline_qty[node] / 20_000.0, 0.0, 5.0)
            edge_features[offset] = feature

        global_features = np.asarray(
            [
                np.clip(self.state.inventory / 50_000.0, 0.0, 5.0),
                np.clip(backlog / 10_000.0, 0.0, 5.0),
                np.clip(pipeline_qty / 50_000.0, 0.0, 5.0),
                np.clip(inbound_1w / 20_000.0, 0.0, 5.0),
                np.clip(inbound_2_3w / 30_000.0, 0.0, 5.0),
                np.clip(current_demand / demand_scale, 0.0, 5.0),
                np.clip(basket_spot / 500.0, 0.0, 2.0),
                policy_summary["supplier_concentration"],
                exogenous_summary["planting_peak"],
                exogenous_summary["monsoon_pressure"],
                exogenous_summary["maintenance_cycle"],
                exogenous_summary["winter_energy"],
                exogenous_summary["supply_warning"],
                exogenous_summary["transport_warning"],
                exogenous_summary["price_warning"],
                np.clip(inbound_4pw / 40_000.0, 0.0, 5.0),
                remaining_weeks_frac,
                week_sin,
                week_cos,
            ],
            dtype=np.float32,
        )

        flat_observation = np.asarray(
            [
                self.state.inventory / 50_000.0,
                backlog / 10_000.0,
                pipeline_qty / 50_000.0,
                inbound_1w / 20_000.0,
                inbound_2_3w / 30_000.0,
                current_demand / demand_scale,
                self.state.cumulative_shortage / 50_000.0,
                basket_spot / 500.0,
                market_summary["max_supply_prob"],
                market_summary["min_supply_multiplier"],
                market_summary["max_node_prob"],
                market_summary["min_node_multiplier"],
                exogenous_summary["planting_peak"],
                exogenous_summary["monsoon_pressure"],
                exogenous_summary["maintenance_cycle"],
                exogenous_summary["winter_energy"],
                exogenous_summary["supply_warning"],
                exogenous_summary["transport_warning"],
                exogenous_summary["price_warning"],
                inbound_4pw / 40_000.0,
                remaining_weeks_frac,
                week_sin,
                week_cos,
                max_route_pipeline_ratio,
                max_supplier_pipeline_ratio,
            ],
            dtype=np.float32,
        )
        global_feature_history = self._update_global_history(global_features)

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_index": EDGE_INDEX.copy(),
            "global_features": global_features,
            "global_feature_history": global_feature_history,
            "flat_observation": flat_observation,
        }
