from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import numpy as np

from decision_strategy import base_policy, basket_price, choose_route_supplier_split
from scenario_tree import build_scenario_tree
from simulation_types import HedgePosition, ModelConfig, Policy, Route, Shipment, State, Supplier
from supply_chain_environment import PRIMARY_ROUTE


ROUTE_ORDER = ("LA_MH", "CA_ZJ", "CA_FCG", "RU_ZJ", "RU_MZL", "JO_LYG", "JO_ZJ")


@dataclass
class TwoStageSPPlan:
    route_quantities: Dict[str, float]
    objective_value: float
    solver_status: str
    route_hedge_ratios: Dict[str, float] | None = None


class _Index:
    def __init__(self, scenario_count: int, horizon: int):
        self.scenario_count = scenario_count
        self.horizon = horizon
        self.route_count = len(ROUTE_ORDER)
        self.x_start = 0
        self.h_start = self.x_start + self.route_count
        self.y_start = self.h_start + self.route_count
        self.i_start = self.y_start + scenario_count * self.route_count
        self.b_start = self.i_start + scenario_count * horizon
        self.e_start = self.b_start + scenario_count * horizon
        self.eta_start = self.e_start + scenario_count * horizon
        self.z_start = self.eta_start + 1
        self.size = self.z_start + scenario_count

    def x(self, route_index: int) -> int:
        return self.x_start + route_index

    def hedge(self, route_index: int) -> int:
        return self.h_start + route_index

    def y(self, scenario_index: int, route_index: int) -> int:
        return self.y_start + scenario_index * self.route_count + route_index

    def inventory(self, scenario_index: int, week_index: int) -> int:
        return self.i_start + scenario_index * self.horizon + week_index

    def backlog(self, scenario_index: int, week_index: int) -> int:
        return self.b_start + scenario_index * self.horizon + week_index

    def emergency(self, scenario_index: int, week_index: int) -> int:
        return self.e_start + scenario_index * self.horizon + week_index

    def eta(self) -> int:
        return self.eta_start

    def z(self, scenario_index: int) -> int:
        return self.z_start + scenario_index


def _existing_arrivals(state: State, absolute_week: int) -> float:
    return float(sum(item.quantity for item in state.shipments if item.arrival_week == absolute_week))


def _forecast_slice(base_demand: np.ndarray, week: int, horizon: int) -> np.ndarray:
    values = base_demand[week: week + horizon]
    if len(values) >= horizon:
        return np.asarray(values, dtype=float)
    tail = float(base_demand[-1]) if len(base_demand) else 0.0
    return np.concatenate([np.asarray(values, dtype=float), np.repeat(tail, horizon - len(values))])


def _route_upper_bound(route: Route, suppliers: Dict[str, Supplier]) -> float:
    supplier = suppliers[route.supplier]
    return float(max(route.base_capacity_tons, supplier.weekly_capacity) * 1.50)


def _fallback_route_quantities(
    state: State,
    week: int,
    routes: Dict[str, Route],
    actual_market: dict,
    base_demand: np.ndarray,
) -> Dict[str, float]:
    policy = base_policy()
    pipeline_qty = sum(item.quantity for item in state.shipments if item.arrival_week > week)
    inventory_position = state.inventory + pipeline_qty
    cover_weeks = int(math.ceil(policy.safety_stock_weeks + 4.0))
    end = min(len(base_demand), week + cover_weeks)
    target_inventory_position = float(base_demand[week:end].sum()) if week < end else float(base_demand[-1])
    planned_order = max(0.0, target_inventory_position - inventory_position)
    route_split = choose_route_supplier_split(policy, actual_market, week)
    route_quantities = {route_name: 0.0 for route_name in ROUTE_ORDER}
    for route_map in route_split.values():
        for route_name, route_share in route_map.items():
            if route_name in routes:
                route_quantities[route_name] += planned_order * max(0.0, float(route_share))
    return route_quantities


def _fallback_hedge_ratios(route_quantities: Dict[str, float]) -> Dict[str, float]:
    return {route_name: (0.30 if route_quantities.get(route_name, 0.0) > 1e-6 else 0.0) for route_name in ROUTE_ORDER}


def _safe_linprog(*args, **kwargs):
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError("two_stage_sp requires scipy.optimize.linprog. Install SciPy or disable this baseline.") from exc
    return linprog(*args, **kwargs)


def solve_two_stage_sp_for_week(
    state: State,
    week: int,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    actual_market: dict,
    base_demand: np.ndarray,
) -> TwoStageSPPlan:
    """Solve a rolling two-stage SAA linear program for current route orders.

    First-stage variables are route order quantities x_r. Scenario-dependent
    recourse variables represent realized usable quantities y_{omega,r},
    inventory, shortage/backlog, and emergency purchases over the lookahead
    horizon. The deterministic equivalent is intentionally compact so it can be
    solved with SciPy during every weekly decision.
    """
    horizon = int(max(1, min(config.lookahead_weeks, config.horizon_weeks - week)))
    scenario_count = int(max(1, config.planning_scenarios))
    current_prices = {code: float(actual_market["spot_prices"][code][week]) for code in suppliers}
    tree = build_scenario_tree(config, suppliers, routes, week, candidate_index=907, start_prices=current_prices)
    paths = tree.paths[:scenario_count]
    index = _Index(len(paths), horizon)

    c = np.zeros(index.size, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None) for _ in range(index.size)]
    bounds[index.eta()] = (None, None)

    cvar_weight = float(np.clip(config.risk_aversion, 0.0, 1.0))
    c[index.eta()] = cvar_weight
    alpha = max(float(config.cvar_alpha), 1e-6)

    for route_index, route_name in enumerate(ROUTE_ORDER):
        route = routes[route_name]
        supplier = suppliers[route.supplier]
        unit_cost = current_prices[route.supplier] + route.base_freight + route.handling_cost
        if route_name != PRIMARY_ROUTE[route.supplier]:
            unit_cost += route.switch_cost
        c[index.x(route_index)] = unit_cost
        bounds[index.x(route_index)] = (0.0, _route_upper_bound(route, suppliers))
        bounds[index.hedge(route_index)] = (0.0, _route_upper_bound(route, suppliers))

        hedge_cost = config.futures_transaction_cost_per_ton
        expected_settlement_spread = 0.0
        for path in paths:
            market = path.market
            delay = int(market["route_delays"][route_name][0])
            arrival_week = route.lead_weeks + delay
            settle_spot = float(market["spot_prices"][route.supplier][min(arrival_week, len(market["spot_prices"][route.supplier]) - 1)])
            locked_price = float(actual_market["futures_prices"][route.supplier][week])
            expected_settlement_spread += float(path.probability) * (settle_spot - locked_price)
        c[index.hedge(route_index)] = hedge_cost + expected_settlement_spread

    for scenario_index, path in enumerate(paths):
        probability = float(path.probability)
        market = path.market
        scenario_demand = np.asarray(market["demand"][:horizon], dtype=float)
        if len(scenario_demand) < horizon:
            fallback = _forecast_slice(base_demand, week, horizon)
            scenario_demand = np.concatenate([scenario_demand, fallback[len(scenario_demand):]])

        for local_week in range(horizon):
            # Shortage is penalized and also loses sales revenue.
            spot_mean = float(np.mean([market["spot_prices"][code][min(local_week, len(market["spot_prices"][code]) - 1)] for code in suppliers]))
            sales_price = config.base_sales_price + 0.20 * (spot_mean - 332.0)
            c[index.inventory(scenario_index, local_week)] = probability * config.storage_cost_per_ton_week
            c[index.backlog(scenario_index, local_week)] = probability * (sales_price + config.shortage_penalty)
            c[index.emergency(scenario_index, local_week)] = probability * (spot_mean + config.emergency_premium)
            emergency_cap = config.emergency_capacity_ratio * (float(scenario_demand[local_week]) + (state.backlog if local_week == 0 else 0.0))
            bounds[index.emergency(scenario_index, local_week)] = (0.0, max(0.0, emergency_cap))
        c[index.z(scenario_index)] = cvar_weight * probability / alpha

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    for scenario_index, path in enumerate(paths):
        market = path.market
        scenario_loss = np.zeros(index.size, dtype=float)
        for route_index, route_name in enumerate(ROUTE_ORDER):
            route = routes[route_name]
            unit_cost = current_prices[route.supplier] + route.base_freight + route.handling_cost
            if route_name != PRIMARY_ROUTE[route.supplier]:
                unit_cost += route.switch_cost
            scenario_loss[index.x(route_index)] = unit_cost

            delay = int(market["route_delays"][route_name][0])
            arrival_week = route.lead_weeks + delay
            settle_spot = float(market["spot_prices"][route.supplier][min(arrival_week, len(market["spot_prices"][route.supplier]) - 1)])
            locked_price = float(actual_market["futures_prices"][route.supplier][week])
            scenario_loss[index.hedge(route_index)] = config.futures_transaction_cost_per_ton + settle_spot - locked_price

            row = np.zeros(index.size, dtype=float)
            row[index.hedge(route_index)] = 1.0
            row[index.x(route_index)] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

            # Nonanticipativity: scenario recourse cannot exceed the route order.
            row = np.zeros(index.size, dtype=float)
            row[index.y(scenario_index, route_index)] = 1.0
            row[index.x(route_index)] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

            local_capacity = (
                route.base_capacity_tons
                * float(market["node_multipliers"][route.node][0])
            )
            row = np.zeros(index.size, dtype=float)
            row[index.y(scenario_index, route_index)] = 1.0
            a_ub.append(row)
            b_ub.append(max(0.0, local_capacity))

        for supplier_code, supplier in suppliers.items():
            row = np.zeros(index.size, dtype=float)
            for route_index, route_name in enumerate(ROUTE_ORDER):
                if routes[route_name].supplier == supplier_code:
                    row[index.y(scenario_index, route_index)] = 1.0
            supply_capacity = supplier.weekly_capacity * float(market["supply_multipliers"][supplier_code][0])
            a_ub.append(row)
            b_ub.append(max(0.0, supply_capacity))

        for local_week in range(horizon):
            spot_mean = float(np.mean([market["spot_prices"][code][min(local_week, len(market["spot_prices"][code]) - 1)] for code in suppliers]))
            sales_price = config.base_sales_price + 0.20 * (spot_mean - 332.0)
            scenario_loss[index.inventory(scenario_index, local_week)] = config.storage_cost_per_ton_week
            scenario_loss[index.backlog(scenario_index, local_week)] = sales_price + config.shortage_penalty
            scenario_loss[index.emergency(scenario_index, local_week)] = spot_mean + config.emergency_premium

            row = np.zeros(index.size, dtype=float)
            row[index.inventory(scenario_index, local_week)] = 1.0
            row[index.backlog(scenario_index, local_week)] = -1.0
            row[index.emergency(scenario_index, local_week)] = -1.0

            if local_week == 0:
                rhs = state.inventory + _existing_arrivals(state, week) - float(market["demand"][0]) - state.backlog
            else:
                row[index.inventory(scenario_index, local_week - 1)] = -1.0
                row[index.backlog(scenario_index, local_week - 1)] = 0.25
                rhs = _existing_arrivals(state, week + local_week) - float(market["demand"][local_week])

            for route_index, route_name in enumerate(ROUTE_ORDER):
                route = routes[route_name]
                delay = int(market["route_delays"][route_name][0])
                arrival_week = route.lead_weeks + delay
                if arrival_week == local_week:
                    row[index.y(scenario_index, route_index)] -= 1.0

            a_eq.append(row)
            b_eq.append(rhs)

        row = scenario_loss
        row[index.eta()] = -1.0
        row[index.z(scenario_index)] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    result = _safe_linprog(
        c,
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        A_eq=np.vstack(a_eq) if a_eq else None,
        b_eq=np.asarray(b_eq, dtype=float) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        route_quantities = _fallback_route_quantities(state, week, routes, actual_market, base_demand)
        return TwoStageSPPlan(
            route_quantities=route_quantities,
            objective_value=float("nan"),
            solver_status=str(result.message),
            route_hedge_ratios=_fallback_hedge_ratios(route_quantities),
        )

    route_quantities = {
        route_name: float(max(0.0, result.x[index.x(route_index)]))
        for route_index, route_name in enumerate(ROUTE_ORDER)
    }
    route_hedge_ratios = {}
    for route_index, route_name in enumerate(ROUTE_ORDER):
        quantity = route_quantities[route_name]
        hedge_qty = float(max(0.0, result.x[index.hedge(route_index)]))
        route_hedge_ratios[route_name] = float(np.clip(hedge_qty / quantity, 0.0, 0.95)) if quantity > 1e-6 else 0.0
    return TwoStageSPPlan(
        route_quantities=route_quantities,
        objective_value=float(result.fun),
        solver_status=str(result.message),
        route_hedge_ratios=route_hedge_ratios,
    )


def route_plan_to_policy(
    route_quantities: Dict[str, float],
    state: State,
    week: int,
    config: ModelConfig,
    routes: Dict[str, Route],
    base_demand: np.ndarray,
    route_hedge_ratios: Dict[str, float] | None = None,
) -> Policy:
    supplier_totals = {code: 0.0 for code in ("LA", "CA", "RU", "JO")}
    for route_name, quantity in route_quantities.items():
        supplier_totals[routes[route_name].supplier] += max(0.0, float(quantity))
    total_quantity = sum(supplier_totals.values())
    if total_quantity <= 1e-6:
        supplier_weights = {"LA": 0.48, "CA": 0.20, "RU": 0.18, "JO": 0.14}
    else:
        supplier_weights = {code: value / total_quantity for code, value in supplier_totals.items()}

    def share(numerator: float, denominator: float) -> float | None:
        if denominator <= 1e-6:
            return None
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    backup_candidates: list[tuple[float, float]] = []
    ca_total = route_quantities.get("CA_ZJ", 0.0) + route_quantities.get("CA_FCG", 0.0)
    ru_total = route_quantities.get("RU_ZJ", 0.0) + route_quantities.get("RU_MZL", 0.0)
    jo_total = route_quantities.get("JO_LYG", 0.0) + route_quantities.get("JO_ZJ", 0.0)
    for value, weight in [
        (share(route_quantities.get("CA_FCG", 0.0), ca_total), ca_total),
        (share(route_quantities.get("RU_MZL", 0.0), ru_total), ru_total),
        (share(2.0 * route_quantities.get("JO_ZJ", 0.0), jo_total), jo_total),
    ]:
        if value is not None and weight > 1e-6:
            backup_candidates.append((value, weight))
    if backup_candidates:
        backup_route_ratio = sum(value * weight for value, weight in backup_candidates) / sum(weight for _, weight in backup_candidates)
    else:
        backup_route_ratio = 0.0

    hedge_candidates = []
    if route_hedge_ratios:
        for route_name, quantity in route_quantities.items():
            if quantity > 1e-6:
                hedge_candidates.append((float(np.clip(route_hedge_ratios.get(route_name, 0.0), 0.0, 0.95)), quantity))
    if hedge_candidates:
        futures_hedge_ratio = sum(value * weight for value, weight in hedge_candidates) / sum(weight for _, weight in hedge_candidates)
    else:
        futures_hedge_ratio = 0.0

    pipeline_qty = sum(item.quantity for item in state.shipments if item.arrival_week > week)
    inventory_position = state.inventory + pipeline_qty
    target_position = inventory_position + total_quantity
    best_safety = 1.0
    best_error = float("inf")
    for safety in np.linspace(1.0, 6.0, 51):
        cover_weeks = int(math.ceil(float(safety) + 4.0))
        end = min(len(base_demand), week + cover_weeks)
        forecast = float(base_demand[week:end].sum()) if week < end else float(base_demand[-1])
        error = abs(forecast - target_position)
        if error < best_error:
            best_error = error
            best_safety = float(safety)

    return Policy(
        supplier_weights=supplier_weights,
        safety_stock_weeks=best_safety,
        reserve_supplier_ratio=0.0,
        backup_route_ratio=float(np.clip(backup_route_ratio, 0.0, 0.8)),
        transport_reserve_ratio=0.0,
        futures_hedge_ratio=float(np.clip(futures_hedge_ratio, 0.0, 0.95)),
    ).normalized()


def apply_two_stage_plan_one_week(
    state: State,
    week: int,
    plan: TwoStageSPPlan,
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    market: dict,
    base_demand: np.ndarray,
) -> Dict[str, float]:
    del base_demand
    arrivals = sum(item.quantity for item in state.shipments if item.arrival_week == week)
    state.shipments = [item for item in state.shipments if item.arrival_week != week]
    state.inventory += arrivals

    proxy_policy = route_plan_to_policy(plan.route_quantities, state, week, config, routes, market["demand"], plan.route_hedge_ratios)
    current_spot = {code: float(market["spot_prices"][code][week]) for code in suppliers}
    current_futures = {code: float(market["futures_prices"][code][week]) for code in suppliers}
    current_basket_spot = basket_price(proxy_policy.supplier_weights, current_spot)

    hedge_pnl = 0.0
    remaining_price_hedges: List[HedgePosition] = []
    for hedge in state.price_hedges:
        if hedge.settle_week == week:
            settle_price = current_basket_spot if hedge.reference == "basket" else current_spot[hedge.reference]
            hedge_pnl += hedge.quantity * (hedge.locked_price - settle_price)
        else:
            remaining_price_hedges.append(hedge)
    state.price_hedges = remaining_price_hedges

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

    supplier_remaining = {
        code: suppliers[code].weekly_capacity * market["supply_multipliers"][code][week]
        for code in suppliers
    }
    procurement_cost = 0.0
    transport_cost = 0.0
    reserve_cost = 0.0
    switch_cost = 0.0
    hedge_cost = 0.0
    procured_qty = 0.0

    for route_name in ROUTE_ORDER:
        requested_qty = max(0.0, float(plan.route_quantities.get(route_name, 0.0)))
        if requested_qty <= 1.0:
            continue
        route = routes[route_name]
        supplier_code = route.supplier
        route_capacity = route.base_capacity_tons * market["node_multipliers"][route.node][week]
        actual_qty = min(requested_qty, supplier_remaining[supplier_code], route_capacity)
        if actual_qty <= 1.0:
            continue

        supplier_remaining[supplier_code] -= actual_qty
        procured_qty += actual_qty
        procurement_cost += actual_qty * current_spot[supplier_code]
        transport_cost += actual_qty * (route.base_freight + route.handling_cost)
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

        hedge_ratio = 0.0 if plan.route_hedge_ratios is None else float(np.clip(plan.route_hedge_ratios.get(route_name, 0.0), 0.0, 0.95))
        price_hedge_qty = actual_qty * hedge_ratio
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
        "policy_safety_weeks": proxy_policy.safety_stock_weeks,
        "policy_backup_ratio": proxy_policy.backup_route_ratio,
        "policy_price_hedge_ratio": proxy_policy.futures_hedge_ratio,
    }
