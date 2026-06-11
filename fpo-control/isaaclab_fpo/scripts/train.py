# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with FPO-RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import viser # HACK: needs to happen *before* Isaac stuff to prevent websockets package error. Super annoying.
# The reason is that Isaac does some crazy path stuff, which will force an older version of websockets to be imported.

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# Hack: wandb sweep doesn't work with "--seed" as a key...
orig_argv = sys.argv.copy()
sys.argv = [arg if not arg.startswith("seed=") else "--" + arg for arg in sys.argv]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with FPO-RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--motion_file", type=str, default=None,
    help="Path to reference motion NPZ file for motion tracking tasks.",
)
# Random masking arguments
parser.add_argument(
    "--use_random_masking",
    action="store_true",
    default=False,
    help="Enable random joint masking during training.",
)
parser.add_argument(
    "--masking_probability",
    type=float,
    default=0.5,
    help="Probability of applying masking to each environment (0.0-1.0).",
)
parser.add_argument(
    "--masking_ratio_min",
    type=float,
    default=0.3,
    help="Minimum ratio of joints to mask when masking is applied (0.0-1.0).",
)
parser.add_argument(
    "--masking_ratio_max",
    type=float,
    default=0.7,
    help="Maximum ratio of joints to mask when masking is applied (0.0-1.0).",
)
parser.add_argument(
    "--joint_masking_prob",
    type=float,
    default=0.8,
    help="Bernoulli probability for masking each selected joint (0.0-1.0).",
)
parser.add_argument(
    "--masking_mode",
    type=str,
    default="random",
    choices=["random", "body_part", "fixed"],
    help="Masking mode: random, body_part, or fixed.",
)
parser.add_argument(
    "--command_joint_indices",
    type=int,
    nargs="*",
    default=None,
    help="Specific joint indices to use as commands (for test time).",
)
parser.add_argument(
    "--command_body_parts",
    type=str,
    nargs="*",
    default=None,
    help="Specific body parts to use as commands (e.g., pelvis left_ankle_roll_link).",
)
# append FPO-RSL-RL cli arguments
cli_args.add_fpo_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for fpo_rsl_rl package."""

# Note: Since fpo_rsl_rl is a local package, we skip version checking
# The package should be available in the project directory

"""Rest everything follows."""

from isaaclab_fpo.patches import apply_isaaclab_patches
apply_isaaclab_patches()

import gymnasium as gym
import os
import torch
from datetime import datetime

from fpo_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401 — registers motion tracking envs
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_fpo.task_cfgs import TASK_CONFIGS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _parse_sweep_overrides():
    """Parse W&B sweep overrides from sys.argv positional args.

    W&B sweeps pass parameters as positional args like:
        agent.policy.actor_hidden_dims=[256,256,256]  agent.algorithm.clip_param=0.05

    Returns two dicts: one for 'agent.*' overrides and one for 'env.*' overrides,
    suitable for passing to configclass.from_dict().
    """
    import ast

    agent_overrides = {}
    env_overrides = {}

    for arg in sys.argv[1:]:
        if "=" not in arg or arg.startswith("-"):
            continue

        key, value_str = arg.split("=", 1)

        # Parse the value
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Handle booleans and strings that ast.literal_eval can't parse
            if value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                value = value_str

        # Route to agent or env overrides
        if key.startswith("agent."):
            parts = key[len("agent."):].split(".")
        elif key.startswith("env."):
            parts = key[len("env."):].split(".")
        else:
            continue

        # Build nested dict from dotted path
        d = agent_overrides if key.startswith("agent.") else env_overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return agent_overrides, env_overrides


def main():
    """Train with FPO-RSL-RL agent."""
    # parse env and agent configs from registries
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_fpo_rsl_rl_cfg(args_cli.task, args_cli)

    # Apply W&B sweep overrides (positional args like agent.policy.actor_hidden_dims=[256,256,256])
    agent_overrides, env_overrides = _parse_sweep_overrides()
    if agent_overrides:
        agent_cfg.from_dict(agent_overrides)
    if env_overrides:
        env_cfg.from_dict(env_overrides)
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # Apply masking configuration if provided via CLI (for non-Hydra usage)
    # Note: When using Hydra overrides, these will already be set in env_cfg
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
        # Only apply CLI args if they were explicitly provided and not using Hydra overrides
        if hasattr(args_cli, 'use_random_masking') and args_cli.use_random_masking:
            env_cfg.commands.motion.use_random_masking = True
            if hasattr(args_cli, 'masking_probability'):
                env_cfg.commands.motion.masking_probability = args_cli.masking_probability
            if hasattr(args_cli, 'masking_ratio_min'):
                env_cfg.commands.motion.masking_ratio_min = args_cli.masking_ratio_min
            if hasattr(args_cli, 'masking_ratio_max'):
                env_cfg.commands.motion.masking_ratio_max = args_cli.masking_ratio_max
            if hasattr(args_cli, 'joint_masking_prob'):
                env_cfg.commands.motion.joint_masking_prob = args_cli.joint_masking_prob
            if hasattr(args_cli, 'masking_mode'):
                env_cfg.commands.motion.masking_mode = args_cli.masking_mode

        if hasattr(args_cli, 'command_joint_indices') and args_cli.command_joint_indices is not None:
            env_cfg.commands.motion.command_joint_indices = args_cli.command_joint_indices

        if hasattr(args_cli, 'command_body_parts') and args_cli.command_body_parts is not None:
            env_cfg.commands.motion.command_body_parts = args_cli.command_body_parts

        if hasattr(args_cli, 'motion_file') and args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
            print(f"[INFO] Using motion file: {env_cfg.commands.motion.motion_file}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "fpo_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_run != '.*':
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        else:
            resume_path = agent_cfg.load_checkpoint

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for fpo-rsl-rl
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from fpo-rsl-rl
    sys.argv = orig_argv  # restore original sys.argv. This will ensure that the wandb run records the correct arguments.
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
