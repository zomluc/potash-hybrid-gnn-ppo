from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from decision_strategy import basket_price, choose_route_supplier_split
from simulation_types import HedgePosition, ModelConfig, Policy, Route, Shipment, State, Supplier
from supply_chain_environment import PRIMARY_ROUTE, clone_state


def forecast_next_weeks(base_demand: np.ndarray, week: int, cover_weeks: int) -> float:
    end = min(len(base_demand), week + cover_weeks)
    if week >= end:
        return float(base_demand[-1])
    return float(base_demand[week:end].sum())


def simulate_one_week(
    state: State,
    week: int,
    policy: Policy,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    market: dict,
    base_demand: np.ndarray,
) -> Dict[str, float]:
    arrivals = sum(item.quantity for item in state.shipments if item.arrival_week == week)
    state.shipments = [item for item in state.shipments if item.arrival_week != week]
    state.inventory += arrivals

    current_spot = {code: float(market["spot_prices"][code][week]) for code in suppliers}
    current_futures = {code: float(market["futures_prices"][code][week]) for code in suppliers}

    hedge_pnl = 0.0
    current_basket_spot = basket_price(policy.supplier_weights, current_spot)

    remaining_price_hedges: List[HedgePosition] = []
    for hedge in state.price_hedges:
        if hedge.settle_week == week:
            settle_price = current_basket_spot if hedge.reference == "basket" else current_spot[hedge.reference]
            hedge_pnl += hedge.quantity * (hedge.locked_price - settle_price)
        else:
            remaining_price_hedges.append(hedge)
    state.price_hedges = remaining_price_hedges

    effective_policy = policy.normalized()

    demand = float(market["demand"][week] + state.backlog)
    state.cumulative_demand += demand
    emergency_capacity = config.emergency_capacity_ratio * demand
    emergency_qty = 0.0

    if state.inventory < demand:
        emergency_qty = min(demand - state.inventory, emergency_capacity)
        state.inventory += emergency_qty

    satisfied = min(state.inventory, demand)
    shortage = max(0.0, demand - satisfied)
    state.backlog = shortage * 0.25
    state.inventory -= satisfied
    state.cumulative_shortage += shortage
    state.cumulative_service += satisfied

    sales_price = config.base_sales_price + 0.20 * (current_basket_spot - 332.0)
    revenue = satisfied * sales_price
    processing_cost = satisfied * config.processing_cost
    emergency_cost = emergency_qty * (current_basket_spot + config.emergency_premium)
    shortage_cost = shortage * config.shortage_penalty
    storage_cost = state.inventory * config.storage_cost_per_ton_week

    average_lead = 4
    cover_weeks = int(math.ceil(effective_policy.safety_stock_weeks + average_lead))
    target_inventory_position = forecast_next_weeks(base_demand, week, cover_weeks)
    pipeline_qty = sum(item.quantity for item in state.shipments if item.arrival_week > week)
    inventory_position = state.inventory + pipeline_qty
    planned_order = max(0.0, target_inventory_position - inventory_position)

    route_split = choose_route_supplier_split(effective_policy, market, week)
    supplier_remaining = {
        code: suppliers[code].weekly_capacity * market["supply_multipliers"][code][week] * (1.0 + 0.7 * effective_policy.reserve_supplier_ratio)
        for code in suppliers
    }
    procurement_cost = 0.0
    transport_cost = 0.0
    reserve_cost = 0.0
    switch_cost = 0.0
    hedge_cost = 0.0
    procured_qty = 0.0

    for supplier_code, route_map in route_split.items():
        for route_name, route_share in route_map.items():
            if route_share <= 0.0:
                continue
            route = routes[route_name]
            requested_qty = planned_order * route_share
            route_capacity = route.base_capacity_tons * market["node_multipliers"][route.node][week] * (1.0 + effective_policy.transport_reserve_ratio)
            actual_qty = min(requested_qty, supplier_remaining[supplier_code], route_capacity)
            if actual_qty <= 1.0:
                continue

            supplier_remaining[supplier_code] -= actual_qty
            procured_qty += actual_qty
            supplier = suppliers[supplier_code]

            purchase_price = current_spot[supplier_code]
            procurement_cost += actual_qty * purchase_price
            transport_cost += actual_qty * (route.base_freight + route.handling_cost)
            reserve_cost += actual_qty * (effective_policy.reserve_supplier_ratio * supplier.reserve_cost + effective_policy.transport_reserve_ratio * route.reserve_cost)
            if route_name != PRIMARY_ROUTE[supplier_code]:
                switch_cost += actual_qty * route.switch_cost

            delay = int(market["route_delays"][route_name][week])
            arrival_week = week + route.lead_weeks + delay
            state.shipments.append(
                Shipment(
                    arrival_week=arrival_week,
                    quantity=actual_qty,
                    supplier_code=supplier_code,
                    route_name=route_name,
                )
            )

            price_hedge_qty = actual_qty * effective_policy.futures_hedge_ratio
            if price_hedge_qty > 0.0 and arrival_week < config.horizon_weeks + config.lookahead_weeks + 6:
                state.price_hedges.append(
                    HedgePosition(
                        settle_week=arrival_week,
                        quantity=price_hedge_qty,
                        locked_price=current_futures[supplier_code],
                        reference=supplier_code,
                    )
                )
                hedge_cost += price_hedge_qty * config.futures_transaction_cost_per_ton

    weekly_profit = revenue - (
        procurement_cost
        + transport_cost
        + reserve_cost
        + switch_cost
        + processing_cost
        + emergency_cost
        + shortage_cost
        + storage_cost
        + hedge_cost
    ) + hedge_pnl
    state.cumulative_profit += weekly_profit

    return {
        "week": week + 1,
        "inventory": state.inventory,
        "backlog": state.backlog,
        "arrivals": arrivals,
        "demand": demand,
        "satisfied": satisfied,
        "shortage": shortage,
        "procured": procured_qty,
        "revenue": revenue,
        "procurement_cost": procurement_cost,
        "transport_cost": transport_cost,
        "reserve_cost": reserve_cost,
        "switch_cost": switch_cost,
        "processing_cost": processing_cost,
        "emergency_cost": emergency_cost,
        "shortage_cost": shortage_cost,
        "storage_cost": storage_cost,
        "hedge_cost": hedge_cost,
        "hedge_pnl": hedge_pnl,
        "weekly_profit": weekly_profit,
        "service_rate": satisfied / demand if demand > 0 else 1.0,
        "policy_safety_weeks": effective_policy.safety_stock_weeks,
        "policy_backup_ratio": effective_policy.backup_route_ratio,
        "policy_price_hedge_ratio": effective_policy.futures_hedge_ratio,
    }


def simulate_horizon(
    start_state: State,
    start_week: int,
    policy: Policy,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    market: dict,
    base_demand: np.ndarray,
    horizon: int,
) -> Dict[str, float]:
    state = clone_state(start_state)
    records = []
    for local_week in range(horizon):
        records.append(simulate_one_week(state, start_week + local_week, policy, config, suppliers, routes, market, base_demand))
    frame = pd.DataFrame(records)
    return {
        "profit": float(frame["weekly_profit"].sum()),
        "service_rate": float(frame["satisfied"].sum() / max(frame["demand"].sum(), 1.0)),
    }