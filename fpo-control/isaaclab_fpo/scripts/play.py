# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from FPO-RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with FPO-RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--testing_phases", type=float, nargs="+", default=None, help="Testing phases for motion tracking (e.g., --testing_phases 0.141 0.013)")
parser.add_argument("--zero_noise_sampling", action="store_true", default=False, help="Use zero noise sampling instead of random noise for flow model (overrides config)")
# Random masking arguments for test time
parser.add_argument(
    "--use_random_masking",
    action="store_true",
    default=False,
    help="Enable random joint masking during testing.",
)
parser.add_argument(
    "--masking_probability",
    type=float,
    default=1.0,
    help="Probability of applying masking to each environment (0.0-1.0). Default 1.0 for testing.",
)
parser.add_argument(
    "--masking_ratio_min",
    type=float,
    default=0.5,
    help="Minimum ratio of joints to mask when masking is applied (0.0-1.0).",
)
parser.add_argument(
    "--masking_ratio_max",
    type=float,
    default=0.5,
    help="Maximum ratio of joints to mask when masking is applied (0.0-1.0).",
)
parser.add_argument(
    "--joint_masking_prob",
    type=float,
    default=1.0,
    help="Bernoulli probability for masking each selected joint (0.0-1.0). Default 1.0 for testing.",
)
parser.add_argument(
    "--masking_mode",
    type=str,
    default="fixed",
    choices=["random", "body_part", "fixed"],
    help="Masking mode: random, body_part, or fixed (recommended for testing).",
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
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from fpo_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with FPO-RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: FpoRslRlOnPolicyRunnerCfg = cli_args.parse_fpo_rsl_rl_cfg(task_name, args_cli)

    # Override testing_phases if provided
    if args_cli.testing_phases is not None and hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
        env_cfg.commands.motion.testing_phases = args_cli.testing_phases
        print(f"[INFO] Using testing phases: {args_cli.testing_phases}")

    # Apply masking configuration if provided
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
        if args_cli.use_random_masking:
            env_cfg.commands.motion.use_random_masking = True
            env_cfg.commands.motion.masking_probability = args_cli.masking_probability
            env_cfg.commands.motion.masking_ratio_min = args_cli.masking_ratio_min
            env_cfg.commands.motion.masking_ratio_max = args_cli.masking_ratio_max
            env_cfg.commands.motion.joint_masking_prob = args_cli.joint_masking_prob
            env_cfg.commands.motion.masking_mode = args_cli.masking_mode
            print(f"[INFO] Using masking - mode: {args_cli.masking_mode}")
            print(f"[INFO]   Masking prob: {args_cli.masking_probability}")
            print(f"[INFO]   Ratio range: [{args_cli.masking_ratio_min}, {args_cli.masking_ratio_max}]")
            print(f"[INFO]   Joint Bernoulli prob: {args_cli.joint_masking_prob}")

        if args_cli.command_joint_indices is not None:
            env_cfg.commands.motion.command_joint_indices = args_cli.command_joint_indices
            print(f"[INFO] Using specific joint indices: {args_cli.command_joint_indices}")

        if args_cli.command_body_parts is not None:
            env_cfg.commands.motion.command_body_parts = args_cli.command_body_parts
            print(f"[INFO] Using specific body parts: {args_cli.command_body_parts}")

    # Override zero_noise_sampling if provided via command line
    if args_cli.zero_noise_sampling:
        agent_cfg.policy.zero_noise_sampling = True
        print(f"[INFO] Overriding config: zero_noise_sampling = True")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "fpo_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("fpo_rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for fpo-rsl-rl
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] Using {'zero' if agent_cfg.policy.zero_noise_sampling else 'random'} noise sampling for flow model")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
