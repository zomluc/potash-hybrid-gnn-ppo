from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from compare_methods import DEFAULT_COMPARISON_SEEDS
from ppo_trainer import load_checkpoint
from rl_environment import PotashSupplyChainEnv
from rl_models import observation_to_torch


CHECKPOINT = Path("results/ppo/ppo_best.pt")
OUTPUT_DIR = Path("results/fusion_gate")


def collect_gate_rows(model: Any, seeds: list[int], device: str) -> pd.DataFrame:
    latest_gate: list[np.ndarray] = []

    def capture_gate(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        latest_gate.append(output.detach().cpu().numpy().reshape(-1))

    handle = model.actor.fusion.gate.register_forward_hook(capture_gate)
    rows: list[dict[str, float | int]] = []
    try:
        model.eval()
        for seed in seeds:
            env = PotashSupplyChainEnv()
            observation, _ = env.reset(seed=seed)
            done = False
            while not done:
                assert env.state is not None
                risk = env.last_info["risk_summary"]
                backlog_before = float(env.state.backlog)
                inventory_before = float(env.state.inventory)
                latest_gate.clear()
                obs_t = observation_to_torch(observation, device=device)
                with torch.no_grad():
                    action, _, _ = model.act(obs_t, deterministic=True)
                if len(latest_gate) != 1:
                    raise RuntimeError(f"Expected one actor-gate output, received {len(latest_gate)}.")
                gate = latest_gate[0]
                observation, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                rows.append(
                    {
                        "seed": int(seed),
                        "week": int(info["week"]),
                        "graph_weight_mean": float(np.mean(gate)),
                        "graph_weight_p10": float(np.quantile(gate, 0.10)),
                        "graph_weight_p50": float(np.quantile(gate, 0.50)),
                        "graph_weight_p90": float(np.quantile(gate, 0.90)),
                        "flat_weight_mean": float(1.0 - np.mean(gate)),
                        "backlog_before": backlog_before,
                        "inventory_before": inventory_before,
                        "supply_risk": float(risk["current_supply_risk"]),
                        "transport_risk": float(risk["current_transport_risk"]),
                        "price_risk": float(risk["current_price_risk"]),
                    }
                )
                done = bool(terminated or truncated)
    finally:
        handle.remove()
    return pd.DataFrame(rows)


def _regime_summary(frame: pd.DataFrame, column: str, high: bool = True) -> dict[str, Any]:
    quantile = 0.75 if high else 0.25
    threshold = float(frame[column].quantile(quantile))
    if frame[column].nunique(dropna=True) < 2:
        return {
            "column": column,
            "focal_regime": "not_identifiable",
            "threshold": threshold,
            "reason": "The observed state variable does not vary across the evaluation sample.",
        }
    selected = frame[column] >= threshold if high else frame[column] <= threshold
    if int(selected.sum()) in {0, len(frame)}:
        return {
            "column": column,
            "focal_regime": "not_identifiable",
            "threshold": threshold,
            "reason": "The requested quantile does not separate the observed evaluation states.",
        }
    focal = frame[selected]
    reference = frame[~selected]
    focal_name = "high" if high else "low"
    return {
        "column": column,
        "focal_regime": focal_name,
        "threshold": threshold,
        "focal_n": int(len(focal)),
        "reference_n": int(len(reference)),
        "focal_graph_weight_mean": float(focal["graph_weight_mean"].mean()),
        "reference_graph_weight_mean": float(reference["graph_weight_mean"].mean()),
        "difference_focal_minus_reference": float(
            focal["graph_weight_mean"].mean() - reference["graph_weight_mean"].mean()
        ),
    }


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    correlations: dict[str, dict[str, float]] = {}
    for column in ["backlog_before", "inventory_before", "supply_risk", "transport_risk", "price_risk"]:
        rho, p_value = spearmanr(frame[column], frame["graph_weight_mean"])
        correlations[column] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    return {
        "checkpoint": str(CHECKPOINT),
        "seeds": [int(seed) for seed in DEFAULT_COMPARISON_SEEDS],
        "observations": int(len(frame)),
        "overall": {
            "graph_weight_mean": float(frame["graph_weight_mean"].mean()),
            "graph_weight_std": float(frame["graph_weight_mean"].std(ddof=1)),
            "flat_weight_mean": float(frame["flat_weight_mean"].mean()),
            "graph_weight_p05": float(frame["graph_weight_mean"].quantile(0.05)),
            "graph_weight_p95": float(frame["graph_weight_mean"].quantile(0.95)),
        },
        "regimes": {
            "high_backlog": _regime_summary(frame, "backlog_before", high=True),
            "low_inventory": _regime_summary(frame, "inventory_before", high=False),
            "high_supply_risk": _regime_summary(frame, "supply_risk", high=True),
            "high_transport_risk": _regime_summary(frame, "transport_risk", high=True),
        },
        "correlations": correlations,
    }


def main() -> None:
    device = "cpu"
    model, _, _ = load_checkpoint(CHECKPOINT, device=device)
    frame = collect_gate_rows(model, list(DEFAULT_COMPARISON_SEEDS), device=device)
    summary = summarize(frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "fusion_gate_by_week.csv", index=False)
    with (OUTPUT_DIR / "fusion_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
