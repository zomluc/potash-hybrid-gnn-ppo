from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from decision_strategy import base_policy, optimize_policy_for_week
from rl_environment import PotashSupplyChainEnv
from rl_models import ActorCriticNetworks, observation_to_torch
from supply_chain_dynamics import simulate_horizon


CHECKPOINT_SCHEMA_VERSION = 8
OBSERVATION_SCHEMA_NAME = "risk_conditioned_gnn_v2"
MODEL_TYPES = {"gnn", "gat", "mlp", "hybrid", "transformer"}


@dataclass
class PPOConfig:
    total_episodes: int = 2000
    rollout_episodes: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 1.8e-4
    critic_lr: float = 4.5e-4
    update_epochs: int = 5
    minibatch_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    actor_hidden_dim: int = 192
    critic_hidden_dim: int = 384
    plain_hidden_dim: int = 256
    num_gnn_layers: int = 3
    gnn_use_temporal_history: bool = True
    gnn_use_attention_pooling: bool = True
    gnn_use_risk_film: bool = True
    gnn_use_edge_enhancement: bool = False
    dropout: float = 0.05
    actor_init_log_std: float = -1.0
    lr_anneal: bool = True
    min_lr_scale: float = 0.4
    device: str = "cpu"
    seed: int = 20260308
    eval_every_episodes: int = 80
    eval_episodes: int = 5
    eval_seeds: List[int] = field(default_factory=lambda: [1101, 1202, 1303, 1404, 1505])
    imitation_warm_start: bool = False
    imitation_episodes: int = 24
    imitation_epochs: int = 8
    imitation_batch_size: int = 256
    imitation_actor_lr: float = 8e-4
    imitation_critic_lr: float = 1.2e-3
    imitation_value_coef: float = 0.5
    imitation_action_mse_coef: float = 0.05
    imitation_seeds: List[int] = field(default_factory=lambda: [2101, 2202, 2303, 2404, 2505, 2606, 2707, 2808])
    model_type: str = "hybrid"
    output_dir: str = "results/ppo"
    checkpoint_every_episodes: int = 80


def recommended_long_run_config(output_dir: str = "results/ppo_3000") -> PPOConfig:
    return PPOConfig(
        total_episodes=3000,
        rollout_episodes=16,
        actor_lr=1.5e-4,
        critic_lr=3.8e-4,
        entropy_coef=0.0025,
        dropout=0.05,
        actor_init_log_std=-1.0,
        lr_anneal=True,
        min_lr_scale=0.2,
        eval_every_episodes=120,
        eval_episodes=5,
        checkpoint_every_episodes=120,
        imitation_warm_start=True,
        imitation_episodes=32,
        imitation_epochs=10,
        imitation_batch_size=256,
        imitation_actor_lr=6e-4,
        imitation_critic_lr=1.0e-3,
        imitation_value_coef=0.4,
        imitation_action_mse_coef=0.05,
        model_type="hybrid",
        gnn_use_temporal_history=True,
        gnn_use_attention_pooling=True,
        gnn_use_risk_film=True,
        gnn_use_edge_enhancement=False,
        output_dir=output_dir,
    )


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: List[dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_value: float,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action.astype(np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.next_values.append(float(next_value))

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values[step] * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages = advantages.astype(np.float32)
        self.returns = returns.astype(np.float32)

    def size(self) -> int:
        return len(self.actions)


def collate_observations(observations: List[dict[str, np.ndarray]], device: torch.device) -> Dict[str, Tensor]:
    batch: Dict[str, Tensor] = {}
    for key in observations[0].keys():
        stacked = np.stack([obs[key] for obs in observations], axis=0)
        dtype = torch.long if key == "edge_index" else torch.float32
        batch[key] = torch.as_tensor(stacked, dtype=dtype, device=device)
    return batch


def clone_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.array(value, copy=True)
        for key, value in observation.items()
    }


def _ensure_batched(observation: dict[str, Tensor]) -> dict[str, Tensor]:
    batch = {}
    for key, value in observation.items():
        if value.dim() == 1:
            batch[key] = value.unsqueeze(0)
        elif key == "edge_index" and value.dim() == 2:
            batch[key] = value.unsqueeze(0)
        elif value.dim() == 2 and key in {"node_features", "edge_features", "global_feature_history"}:
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return batch


def _resolve_model_hidden_dims(config: PPOConfig) -> tuple[int, int]:
    if config.model_type in {"mlp", "transformer"}:
        hidden_dim = int(config.plain_hidden_dim)
        return hidden_dim, hidden_dim
    return int(config.actor_hidden_dim), int(config.critic_hidden_dim)


def evaluate_actions(model: ActorCriticNetworks, observations: dict[str, Tensor], actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    observations = _ensure_batched(observations)
    batch_size = observations["flat_observation"].shape[0]
    log_probs = []
    entropies = []
    values = []

    for idx in range(batch_size):
        obs_i = {key: value[idx] for key, value in observations.items()}
        distribution = model.distribution(obs_i)
        action_i = actions[idx]
        log_probs.append(distribution.log_prob(action_i).sum())
        entropies.append(distribution.entropy().sum())
        values.append(model.value(obs_i))

    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def evaluate_action_statistics(
    model: ActorCriticNetworks,
    observations: dict[str, Tensor],
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    observations = _ensure_batched(observations)
    batch_size = observations["flat_observation"].shape[0]
    log_probs = []
    entropies = []
    values = []
    means = []

    for idx in range(batch_size):
        obs_i = {key: value[idx] for key, value in observations.items()}
        distribution = model.distribution(obs_i)
        action_i = actions[idx]
        log_probs.append(distribution.log_prob(action_i).sum())
        entropies.append(distribution.entropy().sum())
        values.append(model.value(obs_i))
        means.append(distribution.mean)

    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values), torch.stack(means)


def collect_rollout(
    env: PotashSupplyChainEnv,
    model: ActorCriticNetworks,
    config: PPOConfig,
    observation: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[RolloutBuffer, dict[str, np.ndarray], dict[str, Any], int, int]:
    buffer = RolloutBuffer()
    info: dict[str, Any] = {}
    current_observation = observation
    episodes_collected = 0
    steps_collected = 0

    while episodes_collected < config.rollout_episodes:
        obs_t = observation_to_torch(current_observation, device=device)
        with torch.no_grad():
            action_t, log_prob_t, _ = model.act(obs_t, deterministic=False)
            value_t = model.value(obs_t)

        action = action_t.cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs_t = observation_to_torch(next_observation, device=device)
        with torch.no_grad():
            next_value_t = model.value(next_obs_t)

        buffer.add(
            observation=current_observation,
            action=action,
            log_prob=float(log_prob_t.detach().cpu()),
            reward=float(reward),
            done=done,
            value=float(value_t.detach().cpu()),
            next_value=0.0 if done else float(next_value_t.detach().cpu()),
        )
        steps_collected += 1

        if done:
            episodes_collected += 1
            current_observation, _ = env.reset()
        else:
            current_observation = next_observation

    buffer.compute_returns_and_advantages(config.gamma, config.gae_lambda)
    return buffer, current_observation, info, episodes_collected, steps_collected


def ppo_update(
    model: ActorCriticNetworks,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    assert buffer.advantages is not None
    assert buffer.returns is not None

    indices = np.arange(buffer.size())
    actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(np.asarray(buffer.log_probs), dtype=torch.float32, device=device)
    advantages = torch.as_tensor(buffer.advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(buffer.returns, dtype=torch.float32, device=device)

    actor_losses = []
    critic_losses = []
    entropies = []

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, buffer.size(), config.minibatch_size):
            batch_indices = indices[start:start + config.minibatch_size]
            obs_batch = [buffer.observations[idx] for idx in batch_indices]
            action_batch = actions[batch_indices]
            old_log_prob_batch = old_log_probs[batch_indices]
            advantage_batch = advantages[batch_indices]
            return_batch = returns[batch_indices]

            collated_obs = collate_observations(obs_batch, device)
            new_log_probs, entropy, values = evaluate_actions(model, collated_obs, action_batch)
            ratio = (new_log_probs - old_log_prob_batch).exp()

            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            actor_loss = -torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch).mean()
            critic_loss = F.mse_loss(values, return_batch)
            entropy_bonus = entropy.mean()
            total_loss = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy_bonus.detach().cpu()))

    return {
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def evaluate_policy(
    env: PotashSupplyChainEnv,
    model: ActorCriticNetworks,
    episodes: int,
    device: torch.device,
    seeds: List[int] | None = None,
) -> dict[str, float]:
    episode_rewards = []
    episode_total_profits = []
    episode_profits = []

    evaluation_seeds = [int(seed) for seed in seeds] if seeds else [10_000 + episode for episode in range(episodes)]

    for episode_seed in evaluation_seeds:
        observation, _ = env.reset(seed=episode_seed)
        done = False
        total_reward = 0.0
        total_profit = 0.0
        last_profit = 0.0
        while not done:
            obs_t = observation_to_torch(observation, device=device)
            with torch.no_grad():
                action_t, _, _ = model.act(obs_t, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action_t.cpu().numpy())
            total_reward += reward
            total_profit += float(info["metrics"]["weekly_profit"])
            last_profit = info["metrics"]["weekly_profit"]
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_total_profits.append(total_profit)
        episode_profits.append(last_profit)

    return {
        "eval_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "eval_total_profit": float(np.mean(episode_total_profits)) if episode_total_profits else 0.0,
        "eval_last_week_profit": float(np.mean(episode_profits)) if episode_profits else 0.0,
    }


def collect_imitation_dataset(
    env: PotashSupplyChainEnv,
    config: PPOConfig,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray, dict[str, float]]:
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    returns: list[float] = []
    episode_rewards: list[float] = []
    target_episodes = max(1, int(config.imitation_episodes))
    seed_pool = [int(seed) for seed in config.imitation_seeds]
    if len(seed_pool) < target_episodes:
        start = seed_pool[-1] + 101 if seed_pool else 2101
        seed_pool.extend(start + idx for idx in range(target_episodes - len(seed_pool)))

    for episode_index in range(target_episodes):
        episode_seed = seed_pool[episode_index]
        observation, _ = env.reset(seed=episode_seed)
        current_policy = base_policy()
        episode_observations: list[dict[str, np.ndarray]] = []
        episode_actions: list[np.ndarray] = []
        episode_step_rewards: list[float] = []
        done = False

        while not done:
            assert env.state is not None
            assert env.market is not None
            current_policy = optimize_policy_for_week(
                env.state,
                env.week,
                current_policy,
                env.config,
                env.suppliers,
                env.routes,
                env.market,
                env.base_demand,
                simulate_horizon,
            )
            action = env.encode_policy(current_policy)
            episode_observations.append(clone_observation(observation))
            episode_actions.append(action.astype(np.float32))
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_step_rewards.append(float(reward))
            done = terminated or truncated

        discounted_return = 0.0
        episode_returns = [0.0] * len(episode_step_rewards)
        for idx in reversed(range(len(episode_step_rewards))):
            discounted_return = episode_step_rewards[idx] + config.gamma * discounted_return
            episode_returns[idx] = discounted_return

        observations.extend(episode_observations)
        actions.extend(episode_actions)
        returns.extend(episode_returns)
        episode_rewards.append(float(np.sum(episode_step_rewards)))

    summary = {
        "imitation_samples": float(len(actions)),
        "imitation_episodes": float(target_episodes),
        "imitation_mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "imitation_mean_return": float(np.mean(returns)) if returns else 0.0,
    }
    return observations, np.asarray(actions, dtype=np.float32), np.asarray(returns, dtype=np.float32), summary


def _reset_actor_log_std(model: ActorCriticNetworks, value: float) -> None:
    log_std_head = getattr(model.actor, "log_std_head", None)
    if log_std_head is not None and hasattr(log_std_head, "reset_base_log_std"):
        log_std_head.reset_base_log_std(float(value))
        return

    reset_targets = []
    log_std = getattr(model.actor, "log_std", None)
    if log_std is not None:
        reset_targets.append(log_std)
    log_std_head = getattr(model.actor, "log_std_head", None)
    if log_std_head is not None:
        for attribute_name in ["base_log_std", "base_supplier_log_std", "base_control_log_std"]:
            target = getattr(log_std_head, attribute_name, None)
            if target is not None:
                reset_targets.append(target)
    if not reset_targets:
        return
    with torch.no_grad():
        for target in reset_targets:
            target.fill_(float(value))


def run_imitation_warm_start(
    model: ActorCriticNetworks,
    config: PPOConfig,
    device: torch.device,
) -> list[dict[str, float | str]]:
    if not config.imitation_warm_start:
        return []

    print(
        json.dumps(
            {
                "event": "warm_start_start",
                "phase": "imitation",
                "imitation_episodes": int(config.imitation_episodes),
                "imitation_epochs": int(config.imitation_epochs),
                "imitation_batch_size": int(config.imitation_batch_size),
                "imitation_actor_lr": float(config.imitation_actor_lr),
                "imitation_critic_lr": float(config.imitation_critic_lr),
            },
            ensure_ascii=False,
        )
    )

    demo_env = PotashSupplyChainEnv()
    observations, actions, returns, dataset_summary = collect_imitation_dataset(demo_env, config)
    if len(actions) == 0:
        return []

    actor_params = [
        parameter
        for name, parameter in model.actor.named_parameters()
        if "log_std" not in name
    ]
    warm_start_optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": config.imitation_actor_lr},
            {"params": model.critic.parameters(), "lr": config.imitation_critic_lr},
        ]
    )

    history: list[dict[str, float | str]] = []
    indices = np.arange(len(actions))
    model.train()
    for epoch in range(1, max(1, int(config.imitation_epochs)) + 1):
        np.random.shuffle(indices)
        actor_losses = []
        critic_losses = []
        action_mse_values = []
        for start in range(0, len(indices), max(1, int(config.imitation_batch_size))):
            batch_indices = indices[start:start + max(1, int(config.imitation_batch_size))]
            obs_batch = [observations[idx] for idx in batch_indices]
            action_batch = torch.as_tensor(actions[batch_indices], dtype=torch.float32, device=device)
            return_batch = torch.as_tensor(returns[batch_indices], dtype=torch.float32, device=device)

            collated_obs = collate_observations(obs_batch, device)
            log_probs, _, predicted_value, predicted_mean = evaluate_action_statistics(model, collated_obs, action_batch)
            actor_loss = -log_probs.mean()
            action_mse = F.mse_loss(predicted_mean, action_batch)
            critic_loss = F.mse_loss(predicted_value, return_batch)
            total_loss = actor_loss + config.imitation_action_mse_coef * action_mse + config.imitation_value_coef * critic_loss

            warm_start_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            warm_start_optimizer.step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            action_mse_values.append(float(action_mse.detach().cpu()))

        history.append(
            {
                "phase": "imitation",
                "imitation_epoch": float(epoch),
                "episodes_completed": 0.0,
                "imitation_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "imitation_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "imitation_action_mse": float(np.mean(action_mse_values)) if action_mse_values else 0.0,
                **dataset_summary,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))

    _reset_actor_log_std(model, config.actor_init_log_std)
    model.eval()
    print(
        json.dumps(
            {
                "event": "warm_start_complete",
                "phase": "imitation",
                "imitation_epochs": len(history),
                "imitation_samples": int(dataset_summary.get("imitation_samples", 0.0)),
            },
            ensure_ascii=False,
        )
    )
    return history


def train_ppo(config: PPOConfig | None = None) -> tuple[ActorCriticNetworks, list[dict[str, float]]]:
    config = config or PPOConfig()
    if config.model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {config.model_type}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    env = PotashSupplyChainEnv()
    eval_env = PotashSupplyChainEnv()
    observation, _ = env.reset(seed=config.seed)
    actor_hidden_dim, critic_hidden_dim = _resolve_model_hidden_dims(config)
    model = ActorCriticNetworks(
        node_dim=env.node_feature_dim,
        edge_dim=env.edge_feature_dim,
        global_dim=env.global_feature_dim,
        flat_obs_dim=env.flat_feature_dim,
        action_dim=env.action_dim,
        model_type=config.model_type,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        num_gnn_layers=config.num_gnn_layers,
        dropout=config.dropout,
        actor_init_log_std=config.actor_init_log_std,
        gnn_use_temporal_history=config.gnn_use_temporal_history,
        gnn_use_attention_pooling=config.gnn_use_attention_pooling,
        gnn_use_risk_film=config.gnn_use_risk_film,
        gnn_use_edge_enhancement=config.gnn_use_edge_enhancement,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.actor.parameters(), "lr": config.actor_lr},
            {"params": model.critic.parameters(), "lr": config.critic_lr},
        ]
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | str]] = []
    history.extend(run_imitation_warm_start(model, config, device))
    episodes_completed = 0
    update_index = 0
    best_eval_total_profit = float("-inf")
    while episodes_completed < config.total_episodes:
        rollout_episodes = min(config.rollout_episodes, config.total_episodes - episodes_completed)
        rollout_config = PPOConfig(**{**serialize_config(config), "rollout_episodes": rollout_episodes})
        buffer, observation, rollout_info, collected_episodes, collected_steps = collect_rollout(
            env,
            model,
            rollout_config,
            observation,
            device,
        )
        update_index += 1
        episodes_completed += collected_episodes
        if config.lr_anneal:
            progress = min(1.0, episodes_completed / max(config.total_episodes, 1))
            lr_scale = 1.0 - (1.0 - config.min_lr_scale) * progress
            optimizer.param_groups[0]["lr"] = config.actor_lr * lr_scale
            optimizer.param_groups[1]["lr"] = config.critic_lr * lr_scale
        update_stats = ppo_update(model, optimizer, buffer, config, device)
        stats = {
            "phase": "ppo",
            "update": float(update_index),
            "episodes_completed": float(episodes_completed),
            "collected_episodes": float(collected_episodes),
            "collected_steps": float(collected_steps),
            "mean_reward": float(np.mean(buffer.rewards)) if buffer.rewards else 0.0,
            "mean_return": float(np.mean(buffer.returns)) if buffer.returns is not None else 0.0,
            "mean_advantage": float(np.mean(buffer.advantages)) if buffer.advantages is not None else 0.0,
            "mean_value": float(np.mean(buffer.values)) if buffer.values else 0.0,
            "last_week_profit": float(rollout_info.get("metrics", {}).get("weekly_profit", 0.0)),
            "actor_lr": float(optimizer.param_groups[0]["lr"]),
            "critic_lr": float(optimizer.param_groups[1]["lr"]),
            **update_stats,
        }

        should_eval = (
            episodes_completed % config.eval_every_episodes == 0
            or episodes_completed >= config.total_episodes
        )
        if should_eval:
            stats.update(evaluate_policy(eval_env, model, config.eval_episodes, device, seeds=config.eval_seeds))
        history.append(stats)
        print(json.dumps(stats, ensure_ascii=False))

        if should_eval and "eval_total_profit" in stats:
            eval_total_profit = float(stats["eval_total_profit"])
            if eval_total_profit >= best_eval_total_profit:
                best_eval_total_profit = eval_total_profit
                save_best_checkpoint(model, optimizer, config, history, output_dir, step=episodes_completed)

        should_checkpoint = (
            episodes_completed % config.checkpoint_every_episodes == 0
            or episodes_completed >= config.total_episodes
        )
        if should_checkpoint:
            save_checkpoint(model, optimizer, config, history, output_dir, step=episodes_completed)

    return model, history


def serialize_config(config: PPOConfig) -> dict[str, Any]:
    return asdict(config)


def _history_fieldnames(history: list[dict[str, float]]) -> list[str]:
    preferred = [
        "phase",
        "imitation_epoch",
        "imitation_actor_loss",
        "imitation_critic_loss",
        "imitation_action_mse",
        "imitation_samples",
        "imitation_episodes",
        "imitation_mean_episode_reward",
        "imitation_mean_return",
        "update",
        "episodes_completed",
        "collected_episodes",
        "collected_steps",
        "mean_reward",
        "mean_return",
        "mean_advantage",
        "mean_value",
        "last_week_profit",
        "actor_loss",
        "critic_loss",
        "entropy",
        "actor_lr",
        "critic_lr",
        "eval_reward",
        "eval_total_profit",
        "eval_last_week_profit",
    ]
    existing = {key for row in history for key in row.keys()}
    ordered = [name for name in preferred if name in existing]
    extras = sorted(existing.difference(ordered))
    return ordered + extras


def save_training_history_csv(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    fieldnames = _history_fieldnames(history)
    with (output_dir / "ppo_training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize_training_history(history: list[dict[str, float]]) -> dict[str, Any]:
    if not history:
        return {
            "num_updates": 0,
            "episodes_completed_final": 0,
            "best_eval_step": None,
            "best_eval_total_profit": None,
            "final_eval_step": None,
            "final_eval_total_profit": None,
        }

    ppo_history = [row for row in history if row.get("phase", "ppo") != "imitation"]
    imitation_history = [row for row in history if row.get("phase") == "imitation"]
    effective_history = ppo_history if ppo_history else history
    final_row = effective_history[-1]
    eval_rows = [row for row in effective_history if "eval_total_profit" in row]
    best_eval_row = max(eval_rows, key=lambda row: float(row.get("eval_total_profit", float("-inf"))), default=None)
    final_eval_row = eval_rows[-1] if eval_rows else None

    summary = {
        "num_updates": len(history),
        "num_ppo_updates": len(ppo_history),
        "episodes_completed_final": int(final_row.get("episodes_completed", 0)),
        "best_eval_step": int(best_eval_row.get("episodes_completed", 0)) if best_eval_row else None,
        "best_eval_total_profit": float(best_eval_row.get("eval_total_profit")) if best_eval_row else None,
        "best_eval_reward": float(best_eval_row.get("eval_reward")) if best_eval_row and "eval_reward" in best_eval_row else None,
        "final_eval_step": int(final_eval_row.get("episodes_completed", 0)) if final_eval_row else None,
        "final_eval_total_profit": float(final_eval_row.get("eval_total_profit")) if final_eval_row else None,
        "final_eval_reward": float(final_eval_row.get("eval_reward")) if final_eval_row and "eval_reward" in final_eval_row else None,
        "final_actor_lr": float(final_row.get("actor_lr", 0.0)),
        "final_critic_lr": float(final_row.get("critic_lr", 0.0)),
    }
    if imitation_history:
        last_imitation = imitation_history[-1]
        summary.update(
            {
                "imitation_epochs": len(imitation_history),
                "imitation_samples": int(last_imitation.get("imitation_samples", 0)),
                "imitation_final_actor_loss": float(last_imitation.get("imitation_actor_loss", 0.0)),
                "imitation_final_critic_loss": float(last_imitation.get("imitation_critic_loss", 0.0)),
                "imitation_final_action_mse": float(last_imitation.get("imitation_action_mse", 0.0)),
            }
        )
    return summary


def save_training_summary(history: list[dict[str, float]], output_dir: Path) -> None:
    summary = summarize_training_history(history)
    with (output_dir / "ppo_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def _moving_average(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    if window <= 1 or array.size < window:
        return array
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(array, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 220) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def save_training_outputs(history: list[dict[str, float]], config: PPOConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppo_training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with (output_dir / "ppo_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serialize_config(config), handle, ensure_ascii=False, indent=2)
    save_training_history_csv(history, output_dir)
    save_training_summary(history, output_dir)
    save_training_plots(history, output_dir)


def save_training_plots(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    ppo_history = [row for row in history if row.get("phase", "ppo") != "imitation"]
    imitation_history = [row for row in history if row.get("phase") == "imitation"]
    if not ppo_history:
        ppo_history = history

    episodes_completed = [int(item.get("episodes_completed", 0)) for item in ppo_history]
    mean_rewards = [float(item.get("mean_reward", 0.0)) for item in ppo_history]
    mean_returns = [float(item.get("mean_return", 0.0)) for item in ppo_history]
    actor_losses = [float(item.get("actor_loss", 0.0)) for item in ppo_history]
    critic_losses = [float(item.get("critic_loss", 0.0)) for item in ppo_history]
    entropy = [float(item.get("entropy", 0.0)) for item in ppo_history]
    actor_lrs = [float(item.get("actor_lr", 0.0)) for item in ppo_history]
    critic_lrs = [float(item.get("critic_lr", 0.0)) for item in ppo_history]
    eval_rewards = [float(item.get("eval_reward", np.nan)) for item in ppo_history]
    eval_total_profit = [float(item.get("eval_total_profit", np.nan)) for item in ppo_history]
    reward_smooth = _moving_average(mean_rewards, window=5)
    return_smooth = _moving_average(mean_returns, window=5)
    actor_loss_smooth = _moving_average(actor_losses, window=5)
    critic_loss_smooth = _moving_average(critic_losses, window=5)
    entropy_smooth = _moving_average(entropy, window=5)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(episodes_completed, mean_rewards, color="#9ecae1", linewidth=1.2, alpha=0.55, label="Mean Reward")
    axes[0, 0].plot(episodes_completed, reward_smooth, color="#1f77b4", linewidth=2.2, label="Reward (MA-5)")
    axes[0, 0].plot(episodes_completed, mean_returns, color="#9edae5", linewidth=1.2, alpha=0.45, label="Mean Return")
    axes[0, 0].plot(episodes_completed, return_smooth, color="#17becf", linewidth=2.0, label="Return (MA-5)")
    axes[0, 0].set_title("Reward Curve")
    axes[0, 0].set_xlabel("Episodes Completed")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(episodes_completed, actor_losses, color="#98df8a", linewidth=1.2, alpha=0.55)
    axes[0, 1].plot(episodes_completed, actor_loss_smooth, color="#2ca02c", linewidth=2.2)
    axes[0, 1].set_title("Actor Loss Curve")
    axes[0, 1].set_xlabel("Episodes Completed")

    axes[1, 0].plot(episodes_completed, critic_losses, color="#ff9896", linewidth=1.2, alpha=0.55)
    axes[1, 0].plot(episodes_completed, critic_loss_smooth, color="#d62728", linewidth=2.2)
    axes[1, 0].set_title("Critic Loss Curve")
    axes[1, 0].set_xlabel("Episodes Completed")

    axes[1, 1].plot(episodes_completed, entropy, color="#c5b0d5", linewidth=1.2, alpha=0.55, label="Entropy")
    axes[1, 1].plot(episodes_completed, entropy_smooth, color="#9467bd", linewidth=2.2, label="Entropy (MA-5)")
    if not np.all(np.isnan(eval_rewards)):
        axes[1, 1].plot(episodes_completed, eval_rewards, color="#ff7f0e", label="Eval Reward")
    axes[1, 1].set_title("Entropy / Eval Reward")
    axes[1, 1].set_xlabel("Episodes Completed")
    axes[1, 1].legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, output_dir, "ppo_training_curves", dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes_completed, mean_rewards, color="#9ecae1", linewidth=1.2, alpha=0.5, label="Raw")
    ax.plot(episodes_completed, reward_smooth, color="#1f77b4", linewidth=2.4, label="MA-5")
    ax.set_title("PPO Mean Reward")
    ax.set_xlabel("Episodes Completed")
    ax.set_ylabel("Reward")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_dir, "reward_curve", dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes_completed, actor_loss_smooth, color="#2ca02c", linewidth=2.2, label="Actor Loss (MA-5)")
    ax.plot(episodes_completed, critic_loss_smooth, color="#d62728", linewidth=2.2, label="Critic Loss (MA-5)")
    ax.set_title("Actor and Critic Loss")
    ax.set_xlabel("Episodes Completed")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_curves", dpi=260)
    plt.close(fig)

    eval_mask = ~np.isnan(eval_total_profit)
    eval_steps = np.asarray(episodes_completed, dtype=np.int32)[eval_mask]
    eval_profit_values = np.asarray(eval_total_profit, dtype=np.float64)[eval_mask]
    eval_reward_values = np.asarray(eval_rewards, dtype=np.float64)[eval_mask]

    if eval_steps.size > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0, 0].plot(episodes_completed, reward_smooth, color="#1f77b4", linewidth=2.4)
        axes[0, 0].set_title("Training Reward")
        axes[0, 0].set_xlabel("Episodes")
        axes[0, 0].set_ylabel("Mean Reward")

        axes[0, 1].plot(eval_steps, eval_profit_values / 1e6, color="#ff7f0e", marker="o", linewidth=2.2)
        axes[0, 1].set_title("Validation Total Profit")
        axes[0, 1].set_xlabel("Episodes")
        axes[0, 1].set_ylabel("Profit (Million USD)")

        axes[1, 0].plot(episodes_completed, actor_loss_smooth, color="#2ca02c", linewidth=2.2, label="Actor")
        axes[1, 0].plot(episodes_completed, critic_loss_smooth, color="#d62728", linewidth=2.2, label="Critic")
        axes[1, 0].set_title("PPO Losses")
        axes[1, 0].set_xlabel("Episodes")
        axes[1, 0].legend(loc="best")

        axes[1, 1].plot(episodes_completed, actor_lrs, color="#9467bd", linewidth=2.0, label="Actor LR")
        axes[1, 1].plot(episodes_completed, critic_lrs, color="#8c564b", linewidth=2.0, label="Critic LR")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Episodes")
        axes[1, 1].legend(loc="best")

        fig.tight_layout()
        _save_figure(fig, output_dir, "ppo_training_curves_paper", dpi=320)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.plot(eval_steps, eval_profit_values / 1e6, color="#ff7f0e", marker="o", linewidth=2.4, label="Eval total profit")
        best_idx = int(np.argmax(eval_profit_values))
        ax.scatter(eval_steps[best_idx], eval_profit_values[best_idx] / 1e6, color="#d62728", s=48, zorder=3, label="Best checkpoint")
        ax.set_title("Validation Total Profit by Checkpoint")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Profit (Million USD)")
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, output_dir, "eval_total_profit_curve", dpi=320)
        plt.close(fig)

        if not np.all(np.isnan(eval_reward_values)):
            fig, ax = plt.subplots(figsize=(10.5, 5.2))
            ax.plot(eval_steps, eval_reward_values, color="#bcbd22", marker="o", linewidth=2.4)
            ax.set_title("Validation Reward by Checkpoint")
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Eval Reward")
            fig.tight_layout()
            _save_figure(fig, output_dir, "eval_reward_curve", dpi=320)
    plt.close(fig)

    if imitation_history:
        imitation_epochs = [int(item.get("imitation_epoch", 0)) for item in imitation_history]
        imitation_actor_loss = [float(item.get("imitation_actor_loss", 0.0)) for item in imitation_history]
        imitation_critic_loss = [float(item.get("imitation_critic_loss", 0.0)) for item in imitation_history]
        imitation_action_mse = [float(item.get("imitation_action_mse", 0.0)) for item in imitation_history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(imitation_epochs, imitation_actor_loss, color="#1f77b4", marker="o", linewidth=2.2, label="Actor NLL")
        axes[0].plot(imitation_epochs, imitation_action_mse, color="#17becf", marker="o", linewidth=2.2, label="Action MSE")
        axes[0].set_title("Imitation Actor Warm-Start")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(loc="best")
        axes[1].plot(imitation_epochs, imitation_critic_loss, color="#d62728", marker="o", linewidth=2.2)
        axes[1].set_title("Imitation Critic Warm-Start")
        axes[1].set_xlabel("Epoch")
        fig.tight_layout()
        _save_figure(fig, output_dir, "imitation_warm_start_curves", dpi=300)
        plt.close(fig)


def build_actor_critic_from_config(
    env: PotashSupplyChainEnv,
    config: PPOConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> ActorCriticNetworks:
    model_type = "gnn"
    actor_hidden_dim = 128
    critic_hidden_dim = 256
    plain_hidden_dim: int | None = None
    num_gnn_layers = 3
    gnn_use_temporal_history = True
    gnn_use_attention_pooling = True
    gnn_use_risk_film = True
    gnn_use_edge_enhancement = False
    dropout = 0.1
    actor_init_log_std = -0.5
    if isinstance(config, PPOConfig):
        model_type = config.model_type
        actor_hidden_dim = config.actor_hidden_dim
        critic_hidden_dim = config.critic_hidden_dim
        plain_hidden_dim = config.plain_hidden_dim
        num_gnn_layers = config.num_gnn_layers
        gnn_use_temporal_history = config.gnn_use_temporal_history
        gnn_use_attention_pooling = config.gnn_use_attention_pooling
        gnn_use_risk_film = config.gnn_use_risk_film
        gnn_use_edge_enhancement = config.gnn_use_edge_enhancement
        dropout = config.dropout
        actor_init_log_std = config.actor_init_log_std
    elif isinstance(config, dict):
        model_type = str(config.get("model_type", model_type))
        actor_hidden_dim = int(config.get("actor_hidden_dim", actor_hidden_dim))
        critic_hidden_dim = int(config.get("critic_hidden_dim", critic_hidden_dim))
        if "plain_hidden_dim" in config:
            plain_hidden_dim = int(config.get("plain_hidden_dim", actor_hidden_dim))
        num_gnn_layers = int(config.get("num_gnn_layers", num_gnn_layers))
        gnn_use_temporal_history = bool(config.get("gnn_use_temporal_history", gnn_use_temporal_history))
        gnn_use_attention_pooling = bool(config.get("gnn_use_attention_pooling", gnn_use_attention_pooling))
        gnn_use_risk_film = bool(config.get("gnn_use_risk_film", gnn_use_risk_film))
        gnn_use_edge_enhancement = bool(config.get("gnn_use_edge_enhancement", gnn_use_edge_enhancement))
        dropout = float(config.get("dropout", dropout))
        actor_init_log_std = float(config.get("actor_init_log_std", actor_init_log_std))
    if model_type in {"mlp", "transformer"} and plain_hidden_dim is not None:
        actor_hidden_dim = plain_hidden_dim
        critic_hidden_dim = plain_hidden_dim
    model = ActorCriticNetworks(
        node_dim=env.node_feature_dim,
        edge_dim=env.edge_feature_dim,
        global_dim=env.global_feature_dim,
        flat_obs_dim=env.flat_feature_dim,
        action_dim=env.action_dim,
        model_type=model_type,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        actor_init_log_std=actor_init_log_std,
        gnn_use_temporal_history=gnn_use_temporal_history,
        gnn_use_attention_pooling=gnn_use_attention_pooling,
        gnn_use_risk_film=gnn_use_risk_film,
        gnn_use_edge_enhancement=gnn_use_edge_enhancement,
    )
    return model.to(device)


def _checkpoint_env_spec(env: PotashSupplyChainEnv) -> dict[str, Any]:
    return {
        "observation_schema": OBSERVATION_SCHEMA_NAME,
        "node_feature_dim": env.node_feature_dim,
        "edge_feature_dim": env.edge_feature_dim,
        "global_feature_dim": env.global_feature_dim,
        "flat_feature_dim": env.flat_feature_dim,
        "action_dim": env.action_dim,
    }


def _validate_checkpoint_payload(payload: dict[str, Any], env: PotashSupplyChainEnv, checkpoint_path: str | Path) -> None:
    config = payload.get("config", {}) or {}
    model_type = config.get("model_type", "gnn")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Checkpoint {checkpoint_path} uses unsupported model_type '{model_type}'.")

    checkpoint_schema_version = payload.get("checkpoint_schema_version")
    if checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses schema version {checkpoint_schema_version}, expected {CHECKPOINT_SCHEMA_VERSION}. Retrain or load a checkpoint from the observation-enhanced baseline."
        )

    env_spec = payload.get("env_spec")
    if not env_spec:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain environment schema metadata. Retrain under the current schema."
        )

    if env_spec.get("observation_schema") != OBSERVATION_SCHEMA_NAME:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was saved with observation schema '{env_spec.get('observation_schema')}', expected '{OBSERVATION_SCHEMA_NAME}'."
        )

    current_spec = _checkpoint_env_spec(env)
    mismatches = [
        key for key in ["node_feature_dim", "edge_feature_dim", "global_feature_dim", "flat_feature_dim", "action_dim"]
        if env_spec.get(key) != current_spec.get(key)
    ]
    if mismatches:
        details = ", ".join(f"{key}: saved={env_spec.get(key)} current={current_spec.get(key)}" for key in mismatches)
        raise ValueError(
            f"Checkpoint {checkpoint_path} is incompatible with the current environment specification: {details}."
        )


def save_checkpoint(
    model: ActorCriticNetworks,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    history: list[dict[str, float]],
    output_dir: Path,
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"ppo_checkpoint_step_{step}.pt"
    env = PotashSupplyChainEnv()
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": serialize_config(config),
        "env_spec": _checkpoint_env_spec(env),
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    torch.save(payload, output_dir / "ppo_latest.pt")
    return checkpoint_path


def save_best_checkpoint(
    model: ActorCriticNetworks,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    history: list[dict[str, float]],
    output_dir: Path,
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ppo_best.pt"
    env = PotashSupplyChainEnv()
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": serialize_config(config),
        "env_spec": _checkpoint_env_spec(env),
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ActorCriticNetworks, dict[str, Any], PotashSupplyChainEnv]:
    device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=device)
    config_payload = payload.get("config", {})
    env = PotashSupplyChainEnv()
    _validate_checkpoint_payload(payload, env, checkpoint_path)
    model = build_actor_critic_from_config(env, config_payload, device=device)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint {checkpoint_path} could not be loaded into the current actor-critic architecture. Original error: {exc}"
        ) from exc
    model.eval()
    return model, payload, env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for the supply-chain environment.")
    parser.add_argument(
        "--preset",
        choices=["default", "long3000"],
        default="default",
        help="Configuration preset to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the training output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, for example cpu or cuda.",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="Override the number of PPO training episodes.",
    )
    parser.add_argument(
        "--imitation-warm-start",
        action="store_true",
        help="Run heuristic imitation pretraining before PPO updates.",
    )
    parser.add_argument(
        "--imitation-episodes",
        type=int,
        default=None,
        help="Number of heuristic episodes to collect for imitation warm-start.",
    )
    parser.add_argument(
        "--imitation-epochs",
        type=int,
        default=None,
        help="Number of supervised epochs for imitation warm-start.",
    )
    return parser.parse_args()


def build_main_training_config(selected_preset: str = "default") -> PPOConfig:
    total_episodes_in_main = 2000
    imitation_warm_start_in_main = True
    imitation_episodes_in_main = 8
    imitation_epochs_in_main = 6

    config = recommended_long_run_config() if selected_preset == "long3000" else PPOConfig()
    if total_episodes_in_main is not None:
        config.total_episodes = max(1, int(total_episodes_in_main))
    config.imitation_warm_start = bool(imitation_warm_start_in_main)
    if imitation_episodes_in_main is not None:
        config.imitation_episodes = max(1, int(imitation_episodes_in_main))
    if imitation_epochs_in_main is not None:
        config.imitation_epochs = max(1, int(imitation_epochs_in_main))
    return config


def main() -> None:
    args = parse_args()
    selected_preset = args.preset
    output_dir_in_main: str | None = None
    device_in_main: str | None = None

    config = build_main_training_config(selected_preset=selected_preset)
    if output_dir_in_main:
        config.output_dir = output_dir_in_main
    if device_in_main:
        config.device = device_in_main
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.total_episodes is not None:
        config.total_episodes = max(1, int(args.total_episodes))
    if args.imitation_warm_start:
        config.imitation_warm_start = True
    if args.imitation_episodes is not None:
        config.imitation_episodes = max(1, int(args.imitation_episodes))
    if args.imitation_epochs is not None:
        config.imitation_epochs = max(1, int(args.imitation_epochs))
    model, history = train_ppo(config)
    save_training_outputs(history, config, Path(config.output_dir))
    del model


if __name__ == "__main__":
    main()
