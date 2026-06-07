from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from simulation_types import ModelConfig, Route, State, Supplier


PRIMARY_ROUTE = {
    "LA": "LA_MH",
    "CA": "CA_ZJ",
    "RU": "RU_ZJ",
    "JO": "JO_LYG",
}

NODE_BASE_RISK = {
    "ZJ": 0.15,
    "FCG": 0.10,
    "MH": 0.10,
    "LYG": 0.11,
    "MZL": 0.08,
}

NODE_ORDER = tuple(NODE_BASE_RISK.keys())

OBSERVABLE_SIGNAL_ORDER = (
    "mine_accident",
    "sanctions_tightening",
    "maintenance_warning",
    "port_congestion_alert",
    "storm_alert",
    "rail_disruption_alert",
    "energy_shock",
    "demand_surge_signal",
)

SIGNAL_SPECS = {
    "mine_accident": {"base_prob": 0.045, "duration": (2, 4), "amplitude": (0.75, 1.00)},
    "sanctions_tightening": {"base_prob": 0.035, "duration": (3, 6), "amplitude": (0.65, 0.95)},
    "maintenance_warning": {"base_prob": 0.055, "duration": (2, 4), "amplitude": (0.45, 0.80)},
    "port_congestion_alert": {"base_prob": 0.070, "duration": (2, 5), "amplitude": (0.50, 0.90)},
    "storm_alert": {"base_prob": 0.055, "duration": (1, 3), "amplitude": (0.60, 0.95)},
    "rail_disruption_alert": {"base_prob": 0.050, "duration": (2, 4), "amplitude": (0.45, 0.85)},
    "energy_shock": {"base_prob": 0.060, "duration": (2, 5), "amplitude": (0.55, 0.95)},
    "demand_surge_signal": {"base_prob": 0.065, "duration": (3, 6), "amplitude": (0.45, 0.85)},
}

SUPPLIER_SIGNAL_EXPOSURE = {
    "LA": {"mine_accident": 0.25, "maintenance_warning": 0.60, "energy_shock": 0.15},
    "CA": {"mine_accident": 0.75, "maintenance_warning": 0.35, "storm_alert": 0.20, "energy_shock": 0.10},
    "RU": {"sanctions_tightening": 0.90, "maintenance_warning": 0.30, "energy_shock": 0.20},
    "JO": {"port_congestion_alert": 0.50, "storm_alert": 0.55, "maintenance_warning": 0.25, "energy_shock": 0.25},
}

NODE_SIGNAL_EXPOSURE = {
    "ZJ": {"port_congestion_alert": 0.90, "storm_alert": 0.70, "sanctions_tightening": 0.15},
    "FCG": {"port_congestion_alert": 0.75, "storm_alert": 0.60},
    "MH": {"rail_disruption_alert": 0.85, "storm_alert": 0.25},
    "LYG": {"port_congestion_alert": 0.85, "storm_alert": 0.80, "energy_shock": 0.30},
    "MZL": {"port_congestion_alert": 0.40, "storm_alert": 0.45, "sanctions_tightening": 0.30},
}

PRICE_SIGNAL_EXPOSURE = {
    "LA": {"demand_surge_signal": 0.20, "energy_shock": 0.18},
    "CA": {"mine_accident": 0.35, "energy_shock": 0.22, "storm_alert": 0.10},
    "RU": {"sanctions_tightening": 0.40, "energy_shock": 0.20},
    "JO": {"energy_shock": 0.22, "storm_alert": 0.28, "port_congestion_alert": 0.22},
}

DEMAND_SEASONAL_TEMPLATE = np.array([
    1.22, 1.24, 1.28, 1.32, 1.34, 1.33, 1.29, 1.24, 1.18, 1.12,
    1.02, 0.96, 0.92, 0.88, 0.85, 0.83, 0.84, 0.88, 0.98, 1.05,
    1.12, 1.18, 1.22, 1.24, 1.18, 1.10, 0.95, 0.88, 0.82, 0.78,
], dtype=float)


def _resample_profile(values: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if len(values) == horizon:
        return values.astype(float, copy=True)
    source_index = np.linspace(0.0, 1.0, num=len(values), dtype=float)
    target_index = np.linspace(0.0, 1.0, num=horizon, dtype=float)
    return np.interp(target_index, source_index, values).astype(float, copy=False)


def _scaled_window(start: int, end: int, source_horizon: int, target_horizon: int) -> slice:
    start_idx = int(np.floor(start / source_horizon * target_horizon))
    end_idx = int(np.ceil(end / source_horizon * target_horizon))
    start_idx = int(np.clip(start_idx, 0, max(target_horizon - 1, 0)))
    end_idx = int(np.clip(max(end_idx, start_idx + 1), start_idx + 1, target_horizon))
    return slice(start_idx, end_idx)


def _is_in_scaled_window(week: int, start: int, end: int, source_horizon: int, target_horizon: int) -> bool:
    window = _scaled_window(start, end, source_horizon, target_horizon)
    return window.start <= week < window.stop


def build_suppliers(config: ModelConfig | None = None) -> Dict[str, Supplier]:
    supplier_risk_scale = float(config.supplier_risk_scale) if config is not None else 1.0
    return {
        "LA": Supplier("Laos", 1_500_000.0, 315.0, 100.0, float(np.clip(0.04 * supplier_risk_scale, 0.0, 0.95)), 7.0),
        "CA": Supplier("Canada", 2_500_000.0, 360.0, 92.0, float(np.clip(0.08 * supplier_risk_scale, 0.0, 0.95)), 6.0),
        "RU": Supplier("Russia", 3_000_000.0, 330.0, 82.0, float(np.clip(0.10 * supplier_risk_scale, 0.0, 0.95)), 6.5),
        "JO": Supplier("Jordan", 2_000_000.0, 318.0, 84.0, float(np.clip(0.075 * supplier_risk_scale, 0.0, 0.95)), 6.0),
    }


def build_routes() -> Dict[str, Route]:
    return {
        "LA_MH": Route("LA_MH", "LA", "rail", 2, 38.0, 8.0, "MH", 12_000.0, 0.08, 4.0, 5.0),
        "CA_ZJ": Route("CA_ZJ", "CA", "sea", 4, 62.0, 12.0, "ZJ", 10_000.0, 0.12, 4.0, 7.0, backup_route="CA_FCG"),
        "CA_FCG": Route("CA_FCG", "CA", "sea", 5, 70.0, 16.0, "FCG", 6_500.0, 0.08, 4.5, 0.0),
        "RU_ZJ": Route("RU_ZJ", "RU", "sea", 4, 48.0, 12.0, "ZJ", 9_000.0, 0.12, 4.0, 7.0, backup_route="RU_MZL"),
        "RU_MZL": Route("RU_MZL", "RU", "land", 2, 66.0, 10.0, "MZL", 6_500.0, 0.07, 3.5, 0.0),
        "JO_LYG": Route("JO_LYG", "JO", "sea", 4, 58.0, 12.0, "LYG", 8_000.0, 0.18, 5.0, 7.0, backup_route="JO_ZJ"),
        "JO_ZJ": Route("JO_ZJ", "JO", "sea", 5, 66.0, 13.0, "ZJ", 5_500.0, 0.20, 5.0, 0.0),
    }


def build_demand_profile(config: ModelConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    seasonal_weights = _resample_profile(DEMAND_SEASONAL_TEMPLATE, config.horizon_weeks)
    seasonal_weights = seasonal_weights / max(float(seasonal_weights.mean()), 1e-6)
    seasonal_profiles = _seasonal_profiles(config.horizon_weeks)
    peak_window_signal = np.zeros(config.horizon_weeks, dtype=float)
    # Legacy windows are defined on the original 30-week calendar and scaled to the active horizon.
    peak_window_signal[_scaled_window(*config.spring_peak_window_legacy, 30, config.horizon_weeks)] += config.spring_peak_window_boost
    peak_window_signal[_scaled_window(*config.autumn_peak_window_legacy, 30, config.horizon_weeks)] += config.autumn_peak_window_boost
    peak_window_signal[_scaled_window(*config.spring_shoulder_window_legacy, 30, config.horizon_weeks)] += config.spring_shoulder_window_boost
    peak_window_signal[_scaled_window(*config.year_end_support_window_legacy, 30, config.horizon_weeks)] += config.year_end_support_window_boost

    template_intensity = seasonal_weights / max(float(np.max(seasonal_weights)), 1e-6)
    seasonal_intensity = np.clip(
        0.58 * seasonal_profiles["planting_peak"]
        + 0.16 * template_intensity
        + 0.08 * seasonal_profiles["winter_energy"]
        + peak_window_signal,
        0.0,
        1.0,
    )
    centered_intensity = 2.0 * seasonal_intensity - 1.0

    peak_multiplier = max(1.0, float(config.demand_peak_seasonality))
    trough_multiplier = min(1.0, float(config.demand_trough_seasonality))
    positive_intensity = np.clip(centered_intensity, 0.0, None)
    negative_intensity = np.clip(-centered_intensity, 0.0, None)
    positive_branch = 1.0 + (peak_multiplier - 1.0) * np.power(positive_intensity, 1.15)
    negative_branch = 1.0 - (1.0 - trough_multiplier) * np.power(negative_intensity, 0.90)
    seasonal_multiplier = np.where(centered_intensity >= 0.0, positive_branch, negative_branch)
    seasonal_multiplier = np.clip(seasonal_multiplier, trough_multiplier, peak_multiplier)
    base = config.average_weekly_demand_tons * seasonal_multiplier
    if rng is None:
        return base
    shocks = rng.normal(1.0, 0.10, config.horizon_weeks)
    peak_uplift = np.ones(config.horizon_weeks)
    peak_uplift[_scaled_window(*config.spring_peak_window_legacy, 30, config.horizon_weeks)] += rng.uniform(0.12, 0.24)
    peak_uplift[_scaled_window(*config.autumn_peak_window_legacy, 30, config.horizon_weeks)] += rng.uniform(0.10, 0.20)
    peak_uplift[_scaled_window(*config.spring_shoulder_window_legacy, 30, config.horizon_weeks)] += rng.uniform(0.04, 0.10)
    demand = base * shocks * peak_uplift
    return np.maximum(demand, config.average_weekly_demand_tons * max(0.35, trough_multiplier * 0.8))


def generate_price_series(mu: float, sigma: float, phi: float, horizon: int, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(horizon)
    values[0] = mu + rng.normal(0.0, sigma)
    for week in range(1, horizon):
        values[week] = mu + phi * (values[week - 1] - mu) + rng.normal(0.0, sigma)
    return values


def _seasonal_signal_bump(signal_name: str, week: int, horizon: int) -> float:
    if signal_name == "storm_alert":
        return 0.06 if _is_in_scaled_window(week, 16, 26, 30, horizon) else 0.0
    if signal_name == "demand_surge_signal":
        if _is_in_scaled_window(week, 0, 8, 30, horizon) or _is_in_scaled_window(week, 18, 24, 30, horizon):
            return 0.06
        return 0.0
    if signal_name == "maintenance_warning":
        return 0.03 if _is_in_scaled_window(week, 8, 15, 30, horizon) else 0.0
    return 0.0


def _seasonal_profiles(horizon: int) -> dict[str, np.ndarray]:
    weeks = np.arange(horizon, dtype=float)
    phase = 2.0 * np.pi * weeks / max(horizon, 1)

    planting_peak = np.clip(0.55 + 0.45 * np.sin(phase + 0.35), 0.0, 1.0)
    monsoon_pressure = np.clip(0.50 + 0.50 * np.sin(phase - 1.55), 0.0, 1.0)
    maintenance_cycle = np.clip(0.50 + 0.50 * np.sin(phase - 0.55), 0.0, 1.0)
    winter_energy = np.clip(0.50 + 0.50 * np.cos(phase + 0.20), 0.0, 1.0)

    return {
        "planting_peak": planting_peak,
        "monsoon_pressure": monsoon_pressure,
        "maintenance_cycle": maintenance_cycle,
        "winter_energy": winter_energy,
    }


def _generate_signal_series(signal_name: str, horizon: int, rng: np.random.Generator) -> np.ndarray:
    spec = SIGNAL_SPECS[signal_name]
    profiles = _seasonal_profiles(horizon)
    values = np.zeros(horizon, dtype=float)
    remaining = 0
    amplitude = 0.0
    for week in range(horizon):
        if remaining <= 0:
            start_prob = spec["base_prob"] + _seasonal_signal_bump(signal_name, week, horizon)
            if signal_name == "storm_alert":
                start_prob += 0.05 * float(profiles["monsoon_pressure"][week])
            elif signal_name == "demand_surge_signal":
                start_prob += 0.05 * float(profiles["planting_peak"][week])
            elif signal_name == "maintenance_warning":
                start_prob += 0.04 * float(profiles["maintenance_cycle"][week])
            elif signal_name == "energy_shock":
                start_prob += 0.04 * float(profiles["winter_energy"][week])
            if rng.random() < start_prob:
                remaining = int(rng.integers(spec["duration"][0], spec["duration"][1] + 1))
                amplitude = float(rng.uniform(spec["amplitude"][0], spec["amplitude"][1]))
        else:
            if rng.random() < 0.08:
                amplitude = min(1.0, amplitude + float(rng.uniform(0.05, 0.18)))

        if remaining > 0:
            values[week] = float(np.clip(amplitude + rng.normal(0.0, 0.03), 0.0, 1.0))
            remaining -= 1
            amplitude *= float(rng.uniform(0.82, 0.93))
        else:
            values[week] = float(np.clip(rng.normal(0.0, 0.015), 0.0, 0.08))
    return values


def _generate_observable_signals(horizon: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    signals = {name: _generate_signal_series(name, horizon, rng) for name in OBSERVABLE_SIGNAL_ORDER}
    signals["port_congestion_alert"] = np.clip(
        signals["port_congestion_alert"] + 0.20 * signals["storm_alert"],
        0.0,
        1.0,
    )
    signals["rail_disruption_alert"] = np.clip(
        signals["rail_disruption_alert"] + 0.16 * signals["sanctions_tightening"],
        0.0,
        1.0,
    )
    signals["energy_shock"] = np.clip(
        signals["energy_shock"] + 0.12 * signals["sanctions_tightening"],
        0.0,
        1.0,
    )
    return signals


def _lagged_signal(signals: dict[str, np.ndarray], signal_name: str, week: int) -> float:
    if week <= 0:
        return float(0.25 * signals[signal_name][0])
    return float(signals[signal_name][week - 1])


def _build_risk_pressures(horizon: int, signals: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    supply_pressure = np.zeros(horizon, dtype=float)
    transport_pressure = np.zeros(horizon, dtype=float)
    price_pressure = np.zeros(horizon, dtype=float)

    for week in range(horizon):
        prev_supply = float(supply_pressure[week - 1]) if week > 0 else 0.0
        prev_transport = float(transport_pressure[week - 1]) if week > 0 else 0.0
        prev_price = float(price_pressure[week - 1]) if week > 0 else 0.0

        supply_driver = (
            1.10 * _lagged_signal(signals, "mine_accident", week)
            + 0.95 * _lagged_signal(signals, "sanctions_tightening", week)
            + 0.75 * _lagged_signal(signals, "maintenance_warning", week)
            + 0.35 * _lagged_signal(signals, "energy_shock", week)
            + 0.20 * _lagged_signal(signals, "demand_surge_signal", week)
        )
        transport_driver = (
            1.00 * _lagged_signal(signals, "port_congestion_alert", week)
            + 0.80 * _lagged_signal(signals, "storm_alert", week)
            + 0.85 * _lagged_signal(signals, "rail_disruption_alert", week)
            + 0.25 * _lagged_signal(signals, "sanctions_tightening", week)
        )
        price_driver = (
            1.05 * _lagged_signal(signals, "energy_shock", week)
            + 0.90 * _lagged_signal(signals, "demand_surge_signal", week)
            + 0.60 * _lagged_signal(signals, "mine_accident", week)
            + 0.40 * _lagged_signal(signals, "port_congestion_alert", week)
            + 0.35 * _lagged_signal(signals, "sanctions_tightening", week)
            + 0.40 * prev_supply
            + 0.25 * prev_transport
        )

        supply_pressure[week] = float(np.clip(0.45 * prev_supply + 0.35 * np.tanh(supply_driver) + rng.normal(0.0, 0.03), 0.0, 1.0))
        transport_pressure[week] = float(np.clip(0.50 * prev_transport + 0.32 * np.tanh(transport_driver) + rng.normal(0.0, 0.03), 0.0, 1.0))
        price_pressure[week] = float(np.clip(0.42 * prev_price + 0.30 * np.tanh(price_driver) + 0.10 * prev_supply + rng.normal(0.0, 0.035), 0.0, 1.0))

    return {
        "supply": supply_pressure,
        "transport": transport_pressure,
        "price": price_pressure,
    }


def _exposure_value(exposure: dict[str, float], signals: dict[str, np.ndarray], week: int) -> float:
    return float(sum(weight * _lagged_signal(signals, name, week) for name, weight in exposure.items()))


def disruption_state(probability: float, rng: np.random.Generator, stress_level: float = 0.0) -> tuple[float, int, float]:
    probability = float(np.clip(probability + rng.normal(0.0, 0.02), 0.03, 0.85))
    severe = min(0.55, probability * (0.20 + 0.55 * stress_level))
    partial = min(0.92, probability * (0.95 + 0.25 * stress_level))
    draw = rng.random()
    if draw < severe:
        delay = 3 if stress_level < 0.65 else 4
        return 0.18 if stress_level >= 0.55 else 0.24, delay, probability
    if draw < severe + partial:
        delay = 1 if stress_level < 0.45 else 2
        return 0.50 if stress_level >= 0.45 else 0.60, delay, probability
    return 1.0, 0, probability


def generate_market_paths(
    config: ModelConfig,
    suppliers: Dict[str, Supplier],
    routes: Dict[str, Route],
    seed: int,
    horizon: int | None = None,
    start_prices: Dict[str, float] | None = None,
) -> dict:
    horizon = horizon or config.horizon_weeks
    rng = np.random.default_rng(seed)

    seasonal_profiles = _seasonal_profiles(horizon)
    observable_signals = _generate_observable_signals(horizon, rng)
    risk_pressures = _build_risk_pressures(horizon, observable_signals, rng)

    supply_multipliers = {code: np.ones(horizon) for code in suppliers}
    supply_prob = {code: np.zeros(horizon) for code in suppliers}
    node_multipliers = {node: np.ones(horizon) for node in NODE_ORDER}
    node_prob = {node: np.zeros(horizon) for node in NODE_ORDER}
    route_prob = {name: np.zeros(horizon) for name in routes}
    route_delays = {name: np.zeros(horizon, dtype=int) for name in routes}

    spot_prices = {}
    futures_prices = {}
    demand = build_demand_profile(config, rng)
    if horizon != config.horizon_weeks:
        demand = demand[:horizon]
    demand = np.asarray(demand, dtype=float)
    demand *= 1.0 + 0.10 * observable_signals["demand_surge_signal"]

    # Freight variation risk is removed: freight indices are deterministic constants.
    freight_index = {
        "sea": np.ones(horizon, dtype=float),
        "rail": np.ones(horizon, dtype=float),
        "land": np.ones(horizon, dtype=float),
    }
    freight_futures = {
        mode: np.ones(horizon, dtype=float)
        for mode in freight_index
    }

    for week in range(horizon):
        spring_bump = 0.02 + 0.05 * float(seasonal_profiles["planting_peak"][week])
        summer_bump = 0.01 + 0.04 * float(seasonal_profiles["monsoon_pressure"][week])
        maintenance_bump = 0.04 * float(seasonal_profiles["maintenance_cycle"][week])
        energy_bump = 0.04 * float(seasonal_profiles["winter_energy"][week])
        supply_pressure = float(risk_pressures["supply"][week])
        transport_pressure = float(risk_pressures["transport"][week])

        for code, supplier in suppliers.items():
            seasonal_bump = spring_bump if code == "CA" else 0.0
            if code == "LA":
                seasonal_bump = 0.02
            if code in {"CA", "LA"}:
                seasonal_bump += 0.45 * maintenance_bump
            if code == "RU":
                seasonal_bump += 0.35 * energy_bump
            if code == "JO":
                seasonal_bump += 0.30 * summer_bump + 0.20 * energy_bump

            exposure = _exposure_value(SUPPLIER_SIGNAL_EXPOSURE[code], observable_signals, week)
            probability = float(np.clip(supplier.base_risk + seasonal_bump + 0.26 * supply_pressure + 0.18 * exposure, 0.03, 0.82))
            stress_level = float(np.clip(0.65 * supply_pressure + 0.35 * exposure, 0.0, 1.0))
            multiplier, delay, probability = disruption_state(probability, rng, stress_level=stress_level)
            supply_multipliers[code][week] = multiplier
            supply_prob[code][week] = probability
            if delay:
                route_delays[PRIMARY_ROUTE[code]][week] = max(route_delays[PRIMARY_ROUTE[code]][week], delay)

        transport_risk_scale = float(np.clip(config.transport_risk_scale, 0.0, 5.0))
        for node, base_prob in NODE_BASE_RISK.items():
            scaled_base_prob = float(np.clip(base_prob * transport_risk_scale, 0.0, 0.95))
            seasonal_bump = spring_bump if node == "ZJ" else 0.0
            if node in {"ZJ", "FCG"}:
                seasonal_bump += 0.55 * summer_bump
            if node in {"MH", "LYG"}:
                seasonal_bump += 0.40 * maintenance_bump

            exposure = _exposure_value(NODE_SIGNAL_EXPOSURE[node], observable_signals, week)
            probability = float(np.clip(scaled_base_prob + seasonal_bump + summer_bump * 0.7 + 0.28 * transport_pressure + 0.16 * exposure, 0.03, 0.84))
            stress_level = float(np.clip(0.60 * transport_pressure + 0.40 * exposure, 0.0, 1.0))
            multiplier, delay, probability = disruption_state(probability, rng, stress_level=stress_level)
            node_multipliers[node][week] = multiplier
            node_prob[node][week] = probability
            if delay:
                for route in routes.values():
                    if route.node == node:
                        route_delays[route.name][week] = max(route_delays[route.name][week], delay)

        for route in routes.values():
            exposure = _exposure_value(NODE_SIGNAL_EXPOSURE[route.node], observable_signals, week)
            route_probability = float(
                np.clip(
                    route.base_risk * transport_risk_scale + 0.18 * transport_pressure + 0.12 * exposure,
                    0.03,
                    0.84,
                )
            )
            stress_level = float(np.clip(0.60 * transport_pressure + 0.40 * exposure, 0.0, 1.0))
            incremental_probability = max(0.0, route_probability - float(node_prob[route.node][week]))
            _, delay, _ = disruption_state(incremental_probability, rng, stress_level=stress_level)
            route_prob[route.name][week] = max(route_probability, float(node_prob[route.node][week]))
            if delay:
                route_delays[route.name][week] = max(route_delays[route.name][week], delay)

    for code, supplier in suppliers.items():
        exogenous_shock = np.zeros(horizon, dtype=float)
        for week in range(horizon):
            exposure = _exposure_value(PRICE_SIGNAL_EXPOSURE[code], observable_signals, week)
            lagged_supply_prob = float(supply_prob[code][week])
            exogenous_shock[week] = (
                28.0 * risk_pressures["price"][week]
                + 18.0 * risk_pressures["supply"][week]
                + 10.0 * exposure
                + 10.0 * float(seasonal_profiles["winter_energy"][week])
                + 8.0 * float(seasonal_profiles["planting_peak"][week])
                + 14.0 * max(0.0, lagged_supply_prob - supplier.base_risk)
            )

        series = np.zeros(horizon, dtype=float)
        series[0] = supplier.base_fob + exogenous_shock[0] + rng.normal(0.0, 6.0)
        if start_prices and code in start_prices:
            series[0] = start_prices[code]
        for week in range(1, horizon):
            series[week] = (
                supplier.base_fob
                + 0.58 * (series[week - 1] - supplier.base_fob)
                + exogenous_shock[week]
                + rng.normal(0.0, 6.5)
            )
        spot_prices[code] = series
        futures_prices[code] = series + 6.0 + 8.0 * risk_pressures["price"] + rng.normal(0.0, 4.0, horizon)

    return {
        "demand": demand,
        "spot_prices": spot_prices,
        "futures_prices": futures_prices,
        "freight_index": freight_index,
        "freight_futures": freight_futures,
        "supply_multipliers": supply_multipliers,
        "supply_prob": supply_prob,
        "node_multipliers": node_multipliers,
        "node_prob": node_prob,
        "route_prob": route_prob,
        "route_delays": route_delays,
        "observable_signals": observable_signals,
        "risk_pressures": risk_pressures,
        "seasonal_profiles": seasonal_profiles,
    }


def initial_state(config: ModelConfig, base_demand: np.ndarray) -> State:
    return State(inventory=config.initial_inventory_weeks * float(base_demand.mean()))


def clone_state(state: State) -> State:
    return copy.deepcopy(state)
