from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlOnPolicyRunnerCfg


def add_fpo_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add FPO-RSL-RL arguments to the parser."""
    arg_group = parser.add_argument_group("fpo_rsl_rl", description="Arguments for FPO-RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # Self-distillation arguments
    sd_group = parser.add_argument_group("self_distillation", description="Self-distillation pre-training options.")
    sd_group.add_argument(
        "--self_distill", action="store_true", default=False,
        help="Enable self-distillation pre-training phase before FPO.",
    )
    sd_group.add_argument(
        "--self_distill_iterations", type=int, default=None,
        help="Number of self-distillation gradient steps (default: 100).",
    )
    sd_group.add_argument(
        "--self_distill_rollout_steps", type=int, default=None,
        help="Environment steps per distillation data collection round (default: 8).",
    )
    sd_group.add_argument(
        "--self_distill_batch_size", type=int, default=None,
        help="Mini-batch size for distillation gradient steps (default: 16384).",
    )
    sd_group.add_argument(
        "--self_distill_lr", type=float, default=None,
        help="Learning rate for distillation optimizer (default: 3e-4).",
    )
    sd_group.add_argument(
        "--self_distill_n_samples_per_action", type=int, default=None,
        help="CFM samples per action for distillation loss (default: task default).",
    )


def parse_fpo_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> FpoRslRlOnPolicyRunnerCfg:
    """Parse configuration for FPO-RSL-RL agent based on inputs.

    Looks up the task config from the isaaclab_fpo registry instead of gym kwargs.
    """
    from isaaclab_fpo.task_cfgs import TASK_CONFIGS

    if task_name not in TASK_CONFIGS:
        raise KeyError(
            f"No FPO config registered for task '{task_name}'. "
            f"Available tasks: {sorted(TASK_CONFIGS.keys())}"
        )
    agent_cfg = TASK_CONFIGS[task_name]()
    agent_cfg = update_fpo_rsl_rl_cfg(agent_cfg, args_cli)
    return agent_cfg


def update_fpo_rsl_rl_cfg(agent_cfg: FpoRslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for FPO-RSL-RL agent based on inputs."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # Self-distillation CLI overrides
    if getattr(args_cli, "self_distill", False):
        agent_cfg.self_distillation.enabled = True
    if getattr(args_cli, "self_distill_iterations", None) is not None:
        agent_cfg.self_distillation.num_iterations = args_cli.self_distill_iterations
    if getattr(args_cli, "self_distill_rollout_steps", None) is not None:
        agent_cfg.self_distillation.rollout_steps = args_cli.self_distill_rollout_steps
    if getattr(args_cli, "self_distill_batch_size", None) is not None:
        agent_cfg.self_distillation.batch_size = args_cli.self_distill_batch_size
    if getattr(args_cli, "self_distill_lr", None) is not None:
        agent_cfg.self_distillation.learning_rate = args_cli.self_distill_lr
    if getattr(args_cli, "self_distill_n_samples_per_action", None) is not None:
        agent_cfg.self_distillation.n_samples_per_action = args_cli.self_distill_n_samples_per_action

    return agent_cfg
