from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_trainer import load_checkpoint
from rl_models import observation_to_torch


def run_inference(
    checkpoint_path: str | Path = "results/ppo/ppo_latest.pt",
    episodes: int = 3,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    model, payload, env = load_checkpoint(checkpoint_path, device=device)
    episode_summaries = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=20_000 + episode)
        done = False
        total_reward = 0.0
        total_profit = 0.0
        mean_service = []
        while not done:
            obs_t = observation_to_torch(observation, device=device)
            with torch.no_grad():
                action, _, _ = model.act(obs_t, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            total_reward += float(reward)
            total_profit += float(info["metrics"]["weekly_profit"])
            mean_service.append(float(info["metrics"]["service_rate"]))
            done = terminated or truncated

        episode_summaries.append(
            {
                "episode": episode + 1,
                "total_reward": total_reward,
                "total_profit": total_profit,
                "mean_service_rate": float(np.mean(mean_service)) if mean_service else 0.0,
            }
        )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": payload.get("step"),
        "episodes": episode_summaries,
        "mean_total_reward": float(np.mean([item["total_reward"] for item in episode_summaries])) if episode_summaries else 0.0,
        "mean_total_profit": float(np.mean([item["total_profit"] for item in episode_summaries])) if episode_summaries else 0.0,
        "mean_service_rate": float(np.mean([item["mean_service_rate"] for item in episode_summaries])) if episode_summaries else 0.0,
    }
    return summary


def save_inference_plots(summary: dict[str, Any], output_dir: Path) -> None:
    episodes = summary.get("episodes", [])
    if not episodes:
        return

    x = [int(item["episode"]) for item in episodes]
    rewards = [float(item["total_reward"]) for item in episodes]
    profits = [float(item["total_profit"]) for item in episodes]
    service = [float(item["mean_service_rate"]) for item in episodes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(x, rewards, color="#1f77b4")
    axes[0].set_title("Inference Reward by Episode")
    axes[0].set_xlabel("Episode")

    axes[1].bar(x, profits, color="#2ca02c")
    axes[1].set_title("Inference Profit by Episode")
    axes[1].set_xlabel("Episode")

    axes[2].bar(x, service, color="#d62728")
    axes[2].set_title("Service Rate by Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "inference_episode_summary.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, rewards, marker="o", color="#1f77b4", label="Reward")
    ax2 = ax.twinx()
    ax2.plot(x, profits, marker="s", color="#ff7f0e", label="Profit")
    ax.set_title("Inference Reward and Profit")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax2.set_ylabel("Profit")
    fig.tight_layout()
    fig.savefig(output_dir / "inference_reward_profit.png", dpi=220)
    plt.close(fig)


def main() -> None:
    summary = run_inference()
    output_dir = Path("results/ppo")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    save_inference_plots(summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()