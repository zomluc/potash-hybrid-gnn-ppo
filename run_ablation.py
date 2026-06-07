from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

import pandas as pd

from compare_methods import ComparisonConfig, PPOComparisonSpec, evaluate_ppo_episode, run_comparison, save_comparison_outputs
from ppo_trainer import PPOConfig, load_checkpoint, save_training_outputs, train_ppo


@dataclass
class AblationRunConfig:
	label: str
	model_type: str
	output_root: str
	checkpoint_dir: str
	gnn_use_temporal_history: bool = True
	gnn_use_attention_pooling: bool = True
	gnn_use_risk_film: bool = True
	gnn_use_edge_enhancement: bool = True
	total_episodes: int = 1200
	rollout_episodes: int = 16
	eval_every_episodes: int = 80
	eval_episodes: int = 5
	checkpoint_every_episodes: int = 80


@dataclass
class AblationConfig:
	modes: List[str]
	seeds: List[int]
	validation_seeds: List[int]
	total_episodes: int = 1200
	rollout_episodes: int = 16
	eval_every_episodes: int = 80
	eval_episodes: int = 5
	checkpoint_every_episodes: int = 80
	device: str = "cpu"
	seed: int = 20260309
	output_root: str = "results/ablation_hybrid_combo_followup"


def default_ablation_config() -> AblationConfig:
	return AblationConfig(
		modes=[
			"hybrid_gnn_ppo_no_attention_no_edge",
			"hybrid_gnn_ppo_no_attention_no_edge_no_temporal",
		],
		seeds=[101, 202, 303, 404, 505],
		validation_seeds=[1101, 1202, 1303, 1404, 1505],
	)


def _resolve_mode(mode: str) -> dict[str, Any]:
	if mode == "gnn_ppo":
		return {
			"model_type": "gnn",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": True,
		}
	if mode == "hybrid_gnn_ppo":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": True,
		}
	if mode == "hybrid_gnn_ppo_no_attention":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": True,
		}
	if mode == "hybrid_gnn_ppo_no_temporal":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": True,
		}
	if mode == "hybrid_gnn_ppo_no_risk_film":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": True,
		}
	if mode == "hybrid_gnn_ppo_no_edge_enhancement":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "hybrid_gnn_ppo_no_attention_no_edge":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "hybrid_gnn_ppo_no_attention_no_edge_no_temporal":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "hybrid_gnn_ppo_core":
		return {
			"model_type": "hybrid",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "plain_ppo":
		return {
			"model_type": "mlp",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "transformer_ppo":
		return {
			"model_type": "transformer",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "gat_ppo":
		return {
			"model_type": "gat",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "gnn_ppo_core":
		return {
			"model_type": "gnn",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "gnn_ppo_attention_temporal_no_edge":
		return {
			"model_type": "gnn",
			"gnn_use_temporal_history": True,
			"gnn_use_attention_pooling": True,
			"gnn_use_risk_film": False,
			"gnn_use_edge_enhancement": False,
		}
	if mode == "gnn_ppo_risk_film_no_edge":
		return {
			"model_type": "gnn",
			"gnn_use_temporal_history": False,
			"gnn_use_attention_pooling": False,
			"gnn_use_risk_film": True,
			"gnn_use_edge_enhancement": False,
		}
	raise ValueError(f"Unsupported ablation mode: {mode}")


def _mode_dirs(config: AblationConfig, mode: str) -> AblationRunConfig:
	output_root = Path(config.output_root)
	mode_root = output_root / mode
	mode_spec = _resolve_mode(mode)
	return AblationRunConfig(
		label=mode,
		model_type=mode_spec["model_type"],
		output_root=str(mode_root),
		checkpoint_dir=str(mode_root / "ppo"),
		gnn_use_temporal_history=bool(mode_spec["gnn_use_temporal_history"]),
		gnn_use_attention_pooling=bool(mode_spec["gnn_use_attention_pooling"]),
		gnn_use_risk_film=bool(mode_spec["gnn_use_risk_film"]),
		gnn_use_edge_enhancement=bool(mode_spec["gnn_use_edge_enhancement"]),
		total_episodes=config.total_episodes,
		rollout_episodes=config.rollout_episodes,
		eval_every_episodes=config.eval_every_episodes,
		eval_episodes=config.eval_episodes,
		checkpoint_every_episodes=config.checkpoint_every_episodes,
	)


def _checkpoint_step_from_path(checkpoint_path: Path) -> int:
	stem = checkpoint_path.stem
	prefix = "ppo_checkpoint_step_"
	if not stem.startswith(prefix):
		raise ValueError(f"Unexpected checkpoint filename: {checkpoint_path.name}")
	return int(stem[len(prefix):])


def _rank_checkpoints_on_validation(
	config: AblationConfig,
	mode: str,
	checkpoint_dir: Path,
) -> dict[str, Any]:
	checkpoint_paths = sorted(
		checkpoint_dir.glob("ppo_checkpoint_step_*.pt"),
		key=_checkpoint_step_from_path,
	)
	if not checkpoint_paths:
		raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

	validation_rows: list[dict[str, Any]] = []
	best_checkpoint_path: Path | None = None
	best_total_profit_mean = float("-inf")
	best_mean_weekly_reward = float("-inf")

	for checkpoint_path in checkpoint_paths:
		model, _, _ = load_checkpoint(checkpoint_path, device=config.device)
		records = [
			evaluate_ppo_episode(
				model,
				seed,
				device=config.device,
				method_name=mode,
			)
			for seed in config.validation_seeds
		]
		del model

		frame = pd.DataFrame(records)
		total_profit_mean = float(frame["total_profit"].mean()) if not frame.empty else 0.0
		total_profit_std = float(frame["total_profit"].std(ddof=0)) if len(frame) > 1 else 0.0
		mean_weekly_reward_mean = float(frame["mean_weekly_reward"].mean()) if not frame.empty else 0.0
		validation_rows.append(
			{
				"checkpoint_path": str(checkpoint_path),
				"checkpoint_step": _checkpoint_step_from_path(checkpoint_path),
				"validation_total_profit_mean": total_profit_mean,
				"validation_total_profit_std": total_profit_std,
				"validation_mean_weekly_reward_mean": mean_weekly_reward_mean,
			}
		)

		if (
			total_profit_mean > best_total_profit_mean
			or (
				total_profit_mean == best_total_profit_mean
				and mean_weekly_reward_mean > best_mean_weekly_reward
			)
		):
			best_checkpoint_path = checkpoint_path
			best_total_profit_mean = total_profit_mean
			best_mean_weekly_reward = mean_weekly_reward_mean

	if best_checkpoint_path is None:
		raise RuntimeError(f"Unable to select a best checkpoint for mode {mode}")

	validation_rows.sort(key=lambda row: int(row["checkpoint_step"]))
	return {
		"best_checkpoint_path": str(best_checkpoint_path),
		"best_checkpoint_step": _checkpoint_step_from_path(best_checkpoint_path),
		"best_validation_total_profit_mean": best_total_profit_mean,
		"best_validation_mean_weekly_reward_mean": best_mean_weekly_reward,
		"validation_runs": validation_rows,
	}


def train_one_mode(config: AblationConfig, mode: str) -> dict[str, Any]:
	mode_config = _mode_dirs(config, mode)
	ppo_config = PPOConfig(
		total_episodes=mode_config.total_episodes,
		rollout_episodes=mode_config.rollout_episodes,
		eval_every_episodes=mode_config.eval_every_episodes,
		eval_episodes=len(config.validation_seeds),
		eval_seeds=config.validation_seeds,
		checkpoint_every_episodes=mode_config.checkpoint_every_episodes,
		model_type=mode_config.model_type,
		gnn_use_temporal_history=mode_config.gnn_use_temporal_history,
		gnn_use_attention_pooling=mode_config.gnn_use_attention_pooling,
		gnn_use_risk_film=mode_config.gnn_use_risk_film,
		gnn_use_edge_enhancement=mode_config.gnn_use_edge_enhancement,
		output_dir=mode_config.checkpoint_dir,
		device=config.device,
		seed=config.seed,
	)
	model, history = train_ppo(ppo_config)
	output_dir = Path(mode_config.checkpoint_dir)
	save_training_outputs(history, ppo_config, output_dir)
	del model

	latest_checkpoint_path = output_dir / "ppo_latest.pt"
	validation_selection = _rank_checkpoints_on_validation(config, mode, output_dir)
	best_checkpoint_path = Path(validation_selection["best_checkpoint_path"])

	return {
		"mode": mode,
		"model_type": mode_config.model_type,
		"checkpoint_path": str(best_checkpoint_path),
		"latest_checkpoint_path": str(latest_checkpoint_path),
		"training_output_dir": str(output_dir),
		"validation_seeds": config.validation_seeds,
		"history_length": len(history),
		"last_eval_reward": float(history[-1].get("eval_reward", 0.0)) if history else 0.0,
		"last_eval_total_profit": float(history[-1].get("eval_total_profit", 0.0)) if history else 0.0,
		"best_eval_reward": max((float(entry.get("eval_reward", float("-inf"))) for entry in history if entry.get("eval_reward") is not None), default=0.0),
		"best_checkpoint_step": int(validation_selection["best_checkpoint_step"]),
		"best_validation_total_profit_mean": float(validation_selection["best_validation_total_profit_mean"]),
		"best_validation_mean_weekly_reward_mean": float(validation_selection["best_validation_mean_weekly_reward_mean"]),
		"validation_runs": validation_selection["validation_runs"],
	}


def compare_one_mode(config: AblationConfig, mode: str, checkpoint_path: str) -> dict[str, Any]:
	mode_root = Path(config.output_root) / mode
	comparison_dir = mode_root / "comparison"
	mode_spec = _resolve_mode(mode)
	comparison_config = ComparisonConfig(
		seeds=config.seeds,
		output_dir=str(comparison_dir),
		device=config.device,
		ppo_specs=[
			PPOComparisonSpec(
				checkpoint_path=checkpoint_path,
				model_type=mode_spec["model_type"],
				method_label=mode,
				gnn_use_temporal_history=bool(mode_spec["gnn_use_temporal_history"]),
				gnn_use_attention_pooling=bool(mode_spec["gnn_use_attention_pooling"]),
				gnn_use_risk_film=bool(mode_spec["gnn_use_risk_film"]),
				gnn_use_edge_enhancement=bool(mode_spec["gnn_use_edge_enhancement"]),
			)
		],
		include_baseline=False,
	)
	frame, summary = run_comparison(comparison_config)
	save_comparison_outputs(frame, summary, comparison_dir)
	return {
		"mode": mode,
		"comparison_output_dir": str(comparison_dir),
		"summary": summary,
	}


def build_aggregate_summary(comparison_results: List[dict[str, Any]]) -> dict[str, Any]:
	rows = []
	for result in comparison_results:
		mode = result["mode"]
		for item in result["summary"]["summary_by_method"]:
			rows.append({"experiment_mode": mode, **item})
	frame = pd.DataFrame(rows)
	if frame.empty:
		return {"rows": []}
	return {"rows": frame.to_dict(orient="records")}


def save_aggregate_outputs(
	config: AblationConfig,
	training_results: List[dict[str, Any]],
	comparison_results: List[dict[str, Any]],
	aggregate_summary: dict[str, Any],
) -> None:
	output_root = Path(config.output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	with (output_root / "ablation_config.json").open("w", encoding="utf-8") as handle:
		json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
	with (output_root / "training_runs.json").open("w", encoding="utf-8") as handle:
		json.dump(training_results, handle, ensure_ascii=False, indent=2)
	with (output_root / "comparison_runs.json").open("w", encoding="utf-8") as handle:
		json.dump(comparison_results, handle, ensure_ascii=False, indent=2)
	with (output_root / "ablation_summary.json").open("w", encoding="utf-8") as handle:
		json.dump(aggregate_summary, handle, ensure_ascii=False, indent=2)
	rows = aggregate_summary.get("rows", [])
	if rows:
		pd.DataFrame(rows).to_csv(output_root / "ablation_summary.csv", index=False)


def run_ablation(config: AblationConfig | None = None) -> dict[str, Any]:
	config = config or default_ablation_config()
	training_results = []
	comparison_results = []

	for mode in config.modes:
		train_result = train_one_mode(config, mode)
		training_results.append(train_result)
		comparison_results.append(compare_one_mode(config, mode, checkpoint_path=train_result["checkpoint_path"]))

	aggregate_summary = build_aggregate_summary(comparison_results)
	save_aggregate_outputs(config, training_results, comparison_results, aggregate_summary)
	return {
		"training_runs": training_results,
		"comparison_runs": comparison_results,
		"aggregate_summary": aggregate_summary,
	}


def main() -> None:
	result = run_ablation()
	print(json.dumps(result["aggregate_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
