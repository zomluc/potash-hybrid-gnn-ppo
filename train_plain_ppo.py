from __future__ import annotations

from pathlib import Path

from ppo_trainer import build_main_training_config, save_training_outputs, train_ppo


def main() -> None:
    config = build_main_training_config(selected_preset="default")
    config.model_type = "mlp"
    config.output_dir = "results/plain_ppo"
    model, history = train_ppo(config)
    save_training_outputs(history, config, Path(config.output_dir))
    del model


if __name__ == "__main__":
    main()