from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Supplier:
    name: str
    annual_capacity_tons: float
    base_fob: float
    production_cost: float
    base_risk: float
    reserve_cost: float

    @property
    def weekly_capacity(self) -> float:
        return self.annual_capacity_tons / 52.0


@dataclass
class Route:
    name: str
    supplier: str
    mode: str
    lead_weeks: int
    base_freight: float
    handling_cost: float
    node: str
    base_capacity_tons: float
    base_risk: float
    reserve_cost: float
    switch_cost: float
    backup_route: str | None = None


@dataclass
class ModelConfig:
    horizon_weeks: int = 52
    gnn_history_steps: int = 4
    lookahead_weeks: int = 6
    planning_scenarios: int = 24
    risk_aversion: float = 0.65
    cvar_alpha: float = 0.2
    base_sales_price: float = 560.0
    processing_cost: float = 36.0
    storage_cost_per_ton_week: float = 2.4
    shortage_penalty: float = 240.0
    emergency_premium: float = 85.0
    emergency_capacity_ratio: float = 0.05
    futures_transaction_cost_per_ton: float = 0.1
    initial_inventory_weeks: float = 2.0
    average_weekly_demand_tons: float = 12_000.0
    supplier_risk_scale: float = 1.0
    transport_risk_scale: float = 1.0
    demand_peak_seasonality: float = 2.5
    demand_trough_seasonality: float = 0.5
    spring_peak_window_legacy: tuple[int, int] = (1, 11)
    autumn_peak_window_legacy: tuple[int, int] = (16, 28)
    spring_shoulder_window_legacy: tuple[int, int] = (10, 14)
    year_end_support_window_legacy: tuple[int, int] = (28, 30)
    spring_peak_window_boost: float = 0.30
    autumn_peak_window_boost: float = 0.24
    spring_shoulder_window_boost: float = 0.08
    year_end_support_window_boost: float = 0.06


@dataclass
class Policy:
    supplier_weights: Dict[str, float]
    safety_stock_weeks: float
    reserve_supplier_ratio: float
    backup_route_ratio: float
    transport_reserve_ratio: float
    futures_hedge_ratio: float

    def normalized(self) -> "Policy":
        total = sum(max(v, 0.0) for v in self.supplier_weights.values())
        weights = {
            key: (max(value, 0.0) / total if total > 0 else 0.25)
            for key, value in self.supplier_weights.items()
        }
        return Policy(
            supplier_weights=weights,
            safety_stock_weeks=float(np.clip(self.safety_stock_weeks, 1.0, 6.0)),
            reserve_supplier_ratio=float(np.clip(self.reserve_supplier_ratio, 0.0, 0.5)),
            backup_route_ratio=float(np.clip(self.backup_route_ratio, 0.0, 0.8)),
            transport_reserve_ratio=float(np.clip(self.transport_reserve_ratio, 0.0, 0.5)),
            futures_hedge_ratio=float(np.clip(self.futures_hedge_ratio, 0.0, 0.95)),
        )


@dataclass
class Shipment:
    arrival_week: int
    quantity: float
    supplier_code: str = ""
    route_name: str = ""


@dataclass
class HedgePosition:
    settle_week: int
    quantity: float
    locked_price: float
    reference: str


@dataclass
class State:
    inventory: float
    backlog: float = 0.0
    shipments: List[Shipment] = field(default_factory=list)
    price_hedges: List[HedgePosition] = field(default_factory=list)
    cumulative_profit: float = 0.0
    cumulative_service: float = 0.0
    cumulative_demand: float = 0.0
    cumulative_shortage: float = 0.0