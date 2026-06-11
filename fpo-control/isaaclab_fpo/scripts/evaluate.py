# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a checkpoint of an RL agent from FPO-RSL-RL with episode metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import wandb
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with FPO++ and measure episode metrics.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and algorithm. If not specified, uses default from config.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# Evaluation arguments for Motion Tracking task
parser.add_argument("--num_rollouts_per_phase", type=int, default=10, help="Number of rollouts to evaluate per testing phase.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_episode_length", type=int, default=1000, help="Maximum episode length.")
parser.add_argument("--testing_phases", type=float, nargs="+", default=[0.0], help="Testing phases for motion tracking (e.g., --testing_phases 0.141 0.013)")
parser.add_argument("--zero_noise_sampling", action="store_true", default=False, help="Use zero noise sampling instead of random noise for flow model (overrides config)")
parser.add_argument("--training_sampling_steps", type=int, default=None, help="Number of sampling steps for training (default: use training value).")
parser.add_argument("--many_sample_and_average_actions", action="store_true", default=False, help="Use many sample and average actions instead of single action for flow matching inference (overrides config)")
parser.add_argument("--act_n_samples_per_action", type=int, default=50, help="Number of samples per action for many sample and average actions (default: 1).")
parser.add_argument("--flow_sampling_steps", type=int, default=None, help="Number of sampling steps for flow matching inference (default: use training value).")

# W&B checkpoint download arguments
parser.add_argument("--wandb-run-path", type=str, default=None, help="W&B run path (e.g., entity/project/run_id)")
parser.add_argument("--wandb-checkpoint", type=str, default=None, help="W&B checkpoint file name to download (e.g., model_7500.pt)")

# append FPO-RSL-RL cli arguments
cli_args.add_fpo_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch
import copy

from fpo_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_pickle
from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def download_wandb_checkpoint(run_path, checkpoint_name, download_dir="checkpoints"):
    """Download a checkpoint from W&B.

    Args:
        run_path: W&B run path (e.g., "entity/project/run_id")
        checkpoint_name: Name of the checkpoint file to download (e.g., "model_7500.pt")
        download_dir: Directory to download to (default: "checkpoints")

    Returns:
        Path to the downloaded checkpoint file
    """
    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)

    # Create a subdirectory for this specific run
    run_id = run_path.split("/")[-1]
    run_dir = download_path / run_id
    run_dir.mkdir(exist_ok=True)

    # Full path for the checkpoint
    checkpoint_path = run_dir / checkpoint_name

    # Check if checkpoint file already exists locally
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Checkpoint file already exists locally: {checkpoint_path}")
        return str(checkpoint_path)

    # Download from W&B
    print(f"[INFO] Downloading checkpoint from W&B...")
    print(f"       Run path: {run_path}")
    print(f"       Checkpoint: {checkpoint_name}")
    print(f"       Destination: {checkpoint_path}")

    # Initialize W&B API (respects WANDB_BASE_URL env var for custom servers)
    api = wandb.Api()

    # Get the run
    run = api.run(run_path)

    # Download the checkpoint file
    file = run.file(checkpoint_name)
    file.download(root=str(run_dir), replace=True)

    print(f"[INFO] ✅ Downloaded checkpoint to: {checkpoint_path}")

    # Also download params files if they exist
    params_dir = run_dir / "params"
    params_dir.mkdir(exist_ok=True)

    try:
        # Try to download agent.pkl
        agent_file = run.file("params/agent.pkl")
        agent_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] ✅ Downloaded agent config to: {params_dir / 'agent.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download agent.pkl: {e}")

    try:
        # Try to download env.pkl
        env_file = run.file("params/env.pkl")
        env_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] ✅ Downloaded environment config to: {params_dir / 'env.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download env.pkl: {e}")

    return str(checkpoint_path)


def main():
    """Evaluate FPO-RSL-RL agent with episode metrics."""

    ################## Load and override config ##################
    task_name = args_cli.task.split(":")[-1]
    # First determine the checkpoint path
    # Handle W&B checkpoint download if specified
    if args_cli.wandb_run_path and args_cli.wandb_checkpoint:
        resume_path = download_wandb_checkpoint(
            args_cli.wandb_run_path,
            args_cli.wandb_checkpoint
        )
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Parse default config to get experiment name
        agent_cfg_temp = cli_args.parse_fpo_rsl_rl_cfg(task_name, args_cli)
        log_root_path = os.path.join("logs", "fpo_rsl_rl", agent_cfg_temp.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg_temp.load_run, agent_cfg_temp.load_checkpoint)
    
    # Try to load configs from saved params files
    log_dir = os.path.dirname(resume_path)
    agent_pkl_path = os.path.join(log_dir, "params", "agent.pkl")
    env_pkl_path = os.path.join(log_dir, "params", "env.pkl")
    
    # Load agent config
    if os.path.exists(agent_pkl_path):
        print(f"[INFO] Loading agent config from: {agent_pkl_path}")
        agent_cfg = load_pickle(agent_pkl_path)
    else:
        print(f"[WARNING] No saved agent config found at {agent_pkl_path}, using default config")
        agent_cfg: FpoRslRlOnPolicyRunnerCfg = cli_args.parse_fpo_rsl_rl_cfg(task_name, args_cli)
    
    # Load environment config
    if os.path.exists(env_pkl_path):
        print(f"[INFO] Loading environment config from: {env_pkl_path}")
        env_cfg = load_pickle(env_pkl_path)
        print(f"[INFO] Loaded config has num_envs: {env_cfg.scene.num_envs}. Overriding to {args_cli.num_envs}.")
        # Override some settings for playback
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
    else:
        print(f"[WARNING] No saved environment config found at {env_pkl_path}, using default config")
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        print(f"[INFO] Created environment config with num_envs: {env_cfg.scene.num_envs}")
    
    # Don't move this. This should be before the environment creation.
    print(f"[INFO] Using seed from command line: {args_cli.seed} for environment config.")
    env_cfg.seed = args_cli.seed
    
    # Override termination thresholds for evaluation; 
    # Loose the termination criteria and just keep the timeout termination.
    env_cfg.terminations.ref_pos.params["threshold"] = 1000
    env_cfg.terminations.ref_ori.params["threshold"] = 1000
    env_cfg.terminations.ee_body_pos.params["threshold"] = 1000

    print(f"[INFO] Loading experiment from directory: {log_dir}")
    print(f"[INFO] Using policy network hidden dims: {agent_cfg.policy.actor_hidden_dims}")
    
    # Override testing phases for motion tracking if specified
    if args_cli.testing_phases is not None:
        # Check if this is a motion tracking task and set the testing phases
        if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
            if type(env_cfg.commands.motion).__name__ == 'MotionCommandCfg':
                print(f"[INFO] Setting motion tracking testing phases to: {args_cli.testing_phases}")
                env_cfg.commands.motion.testing_phases = args_cli.testing_phases
    
    # Override zero_noise_sampling if provided via command line
    if args_cli.many_sample_and_average_actions and hasattr(agent_cfg.policy, 'many_sample_and_average_actions'):
        agent_cfg.policy.many_sample_and_average_actions = True
        agent_cfg.policy.act_n_samples_per_action = args_cli.act_n_samples_per_action 
        agent_cfg.policy.zero_noise_sampling = False
        print(f"[INFO] Overriding config: many_sample_and_average_actions = True, zero_noise_sampling = False, act_n_samples_per_action = {args_cli.act_n_samples_per_action}")
    elif args_cli.zero_noise_sampling and hasattr(agent_cfg.policy, 'zero_noise_sampling'):
        agent_cfg.policy.zero_noise_sampling = True
        print(f"[INFO] Overriding config: zero_noise_sampling = True, many_sample_and_average_actions = False")

    # Override the training sample steps if it is not None
    if args_cli.training_sampling_steps is not None and hasattr(agent_cfg.policy, 'training_sampling_steps'):
        print(f"[INFO] Overriding training sampling steps from {ppo_runner.alg.policy.training_sampling_steps} to {args_cli.training_sampling_steps}")
        agent_cfg.policy.training_sampling_steps = args_cli.training_sampling_steps
    elif hasattr(agent_cfg.policy, 'training_sampling_steps'):
        print(f"[INFO] Using training sampling steps: {agent_cfg.policy.training_sampling_steps}")

    # Override flow sampling steps if specified
    if args_cli.flow_sampling_steps is not None and hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Overriding flow sampling steps from {agent_cfg.policy.sampling_steps} to {args_cli.flow_sampling_steps}")
        agent_cfg.policy.sampling_steps = args_cli.flow_sampling_steps
    elif hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Using flow sampling steps: {agent_cfg.policy.sampling_steps}")

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for fpo-rsl-rl
    env = FpoRslRlVecEnvWrapper(env, clip_actions=None)

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Apply the sampling steps override to the loaded policy if it was changed
    if args_cli.flow_sampling_steps is not None and hasattr(ppo_runner.alg.policy, 'sampling_steps'):
        ppo_runner.alg.policy.sampling_steps = args_cli.flow_sampling_steps

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs

    # Calculate total episodes to run
    total_episodes = len(args_cli.testing_phases) * args_cli.num_rollouts_per_phase

    # Initialize metrics tracking - overall and per phase
    episode_rewards = []
    episode_lengths = []
    phase_metrics = {phase: {"rewards": [], "lengths": []} for phase in args_cli.testing_phases}

    current_episode_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    current_episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=env.unwrapped.device)

    # Track which phase each environment is currently running
    # This will be updated based on what the environment actually samples
    current_episode_phases = {}

    episodes_completed = 0
    phase_counts = {phase: 0 for phase in args_cli.testing_phases}

    # reset environment
    obs, _ = env.get_observations()

    # Get initial phases that were sampled
    if hasattr(env.unwrapped, 'command_manager'):
        motion_command = env.unwrapped.command_manager._terms["motion"]
        for env_id in range(num_envs):
            # Get the actual phase that was sampled for this environment
            time_step = motion_command.time_steps[env_id].item()
            phase = time_step / max(motion_command.motion.time_step_total - 1, 1)
            # Find closest testing phase
            closest_phase = min(args_cli.testing_phases, key=lambda x: abs(x - phase))
            current_episode_phases[env_id] = closest_phase

    timestep = 0

    print(f"\n[INFO] Starting evaluation")
    print(f"[INFO] Testing phases: {args_cli.testing_phases}")
    print(f"[INFO] Rollouts per phase: {args_cli.num_rollouts_per_phase}")
    print(f"[INFO] Total episodes target: {total_episodes}")
    print(f"[INFO] Number of parallel environments: {num_envs}")
    print("-" * 50)

    # Run until we have enough samples for each phase
    while simulation_app.is_running():
        # Check if we have enough samples for all phases
        all_phases_complete = all(
            phase_counts[phase] >= args_cli.num_rollouts_per_phase
            for phase in args_cli.testing_phases
        )
        if all_phases_complete:
            break

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            # Update episode metrics
            current_episode_rewards += rewards
            current_episode_lengths += 1

            # Check for completed episodes
            done_indices = torch.where(dones)[0]
            if len(done_indices) > 0:

                for idx in done_indices:
                    idx_item = idx.item()

                    # Get the phase for this episode
                    phase = current_episode_phases.get(idx_item, 0.0)

                    # Only record if we still need more samples for this phase
                    # Hongsuk - I think this is some asynchronous issue in simulation env.
                    if current_episode_lengths[idx].item() <= 1:
                        pass
                    
                    elif phase in phase_counts and phase_counts[phase] < args_cli.num_rollouts_per_phase:
                        # Store completed episode metrics
                        reward = current_episode_rewards[idx].item()
                        length = current_episode_lengths[idx].item()

                        episode_rewards.append(reward)
                        episode_lengths.append(length)

                        # Store per-phase metrics
                        phase_metrics[phase]["rewards"].append(reward)
                        phase_metrics[phase]["lengths"].append(length)
                        phase_counts[phase] += 1

                        episodes_completed += 1

                        # Print progress
                        print(f"Phase {phase:.3f}: {phase_counts[phase]}/{args_cli.num_rollouts_per_phase} | "
                              f"Reward: {reward:.2f}, Length: {length:.0f}")

                    # Reset metrics for completed environment
                    current_episode_rewards[idx] = 0
                    current_episode_lengths[idx] = 0

                    # After reset, get the new phase that was sampled
                    # This will happen on the next step when the environment resets

            # Check max episode length
            max_length_reached = current_episode_lengths >= args_cli.max_episode_length
            if max_length_reached.any():
                for idx in torch.where(max_length_reached)[0]:
                    idx_item = idx.item()

                    # Get the phase for this episode
                    phase = current_episode_phases.get(idx_item, 0.0)

                    # Only record if we still need more samples for this phase
                    if phase in phase_counts and phase_counts[phase] < args_cli.num_rollouts_per_phase:
                        # Store completed episode metrics
                        reward = current_episode_rewards[idx].item()
                        length = current_episode_lengths[idx].item()

                        episode_rewards.append(reward)
                        episode_lengths.append(length)

                        # Store per-phase metrics
                        phase_metrics[phase]["rewards"].append(reward)
                        phase_metrics[phase]["lengths"].append(length)
                        phase_counts[phase] += 1

                        episodes_completed += 1
                        print(f"[WARNING] Max length - Phase {phase:.3f}: {phase_counts[phase]}/{args_cli.num_rollouts_per_phase} | "
                              f"Reward: {reward:.2f}, Length: {length:.0f}")

                    current_episode_rewards[idx] = 0
                    current_episode_lengths[idx] = 0

        # Update phase tracking after environment steps
        # The environment will have reset any done environments
        if len(done_indices) > 0 or max_length_reached.any():
            if hasattr(env.unwrapped, 'command_manager'):
                motion_command = env.unwrapped.command_manager._terms["motion"]
                for idx in torch.cat([done_indices, torch.where(max_length_reached)[0]]):
                    idx_item = idx.item()
                    # Get the actual phase that was sampled for this environment after reset
                    time_step = motion_command.time_steps[idx_item].item()
                    phase = time_step / max(motion_command.motion.time_step_total - 1, 1)
                    # Find closest testing phase
                    closest_phase = min(args_cli.testing_phases, key=lambda x: abs(x - phase))
                    current_episode_phases[idx_item] = closest_phase

        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Calculate and display final metrics
    if len(episode_rewards) > 0:
        print("\n" + "=" * 60)
        print("OVERALL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total episodes evaluated: {len(episode_rewards)}")
        print(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Min/Max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
        print(f"Min/Max length: {np.min(episode_lengths):.0f} / {np.max(episode_lengths):.0f}")

        print("\n" + "-" * 60)
        print("PER-PHASE RESULTS")
        print("-" * 60)
        print(f"{'Phase':>10} | {'Rollouts':>8} | {'Mean Reward':>12} | {'Std Reward':>10} | {'Mean Length':>11} | {'Std Length':>10}")
        print("-" * 60)

        for phase in sorted(args_cli.testing_phases):
            if phase in phase_metrics and len(phase_metrics[phase]["rewards"]) > 0:
                rewards = phase_metrics[phase]["rewards"]
                lengths = phase_metrics[phase]["lengths"]
                print(f"{phase:10.3f} | {len(rewards):8d} | "
                      f"{np.mean(rewards):12.2f} | {np.std(rewards):10.2f} | "
                      f"{np.mean(lengths):11.1f} | {np.std(lengths):10.1f}")

        print("=" * 60)

        # Save detailed metrics to file
        metrics_file = os.path.join(log_dir, f"evaluation_metrics_zerosampling_{agent_cfg.policy.zero_noise_sampling}_flowsampling_{agent_cfg.policy.sampling_steps}_manysample_{agent_cfg.policy.many_sample_and_average_actions}_actn_{agent_cfg.policy.act_n_samples_per_action}.txt")
        with open(metrics_file, "w", encoding="utf-8") as f:    
            f.write(f"Phase-based Evaluation Results\n")
            f.write(f"==============================\n")
            f.write(f"Checkpoint: {resume_path}\n")
            f.write(f"Testing phases: {args_cli.testing_phases}\n")
            f.write(f"Rollouts per phase: {args_cli.num_rollouts_per_phase}\n")
            f.write(f"Total episodes: {len(episode_rewards)}\n")
            f.write(f"\nOVERALL METRICS:\n")
            f.write(f"-----------------\n")
            f.write(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}\n")
            f.write(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}\n")
            f.write(f"Min/Max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}\n")
            f.write(f"Min/Max length: {np.min(episode_lengths):.0f} / {np.max(episode_lengths):.0f}\n")

            f.write(f"\nPER-PHASE METRICS:\n")
            f.write(f"------------------\n")
            for phase in sorted(args_cli.testing_phases):
                if phase in phase_metrics and len(phase_metrics[phase]["rewards"]) > 0:
                    rewards = phase_metrics[phase]["rewards"]
                    lengths = phase_metrics[phase]["lengths"]
                    f.write(f"\nPhase {phase:.3f}:\n")
                    f.write(f"  Rollouts completed: {len(rewards)}\n")
                    f.write(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
                    f.write(f"  Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}\n")
                    f.write(f"  Min/Max reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}\n")
                    f.write(f"  Min/Max length: {np.min(lengths):.0f} / {np.max(lengths):.0f}\n")

        print(f"\n[INFO] Detailed metrics saved to: {metrics_file}")

        # Save raw data for further analysis
        import json
        raw_data_file = os.path.join(log_dir, f"evaluation_raw_data_zerosampling_{agent_cfg.policy.zero_noise_sampling}_flowsampling_{agent_cfg.policy.sampling_steps}_manysample_{agent_cfg.policy.many_sample_and_average_actions}_actn_{agent_cfg.policy.act_n_samples_per_action}.json")
        raw_data = {
            "testing_phases": args_cli.testing_phases,
            "rollouts_per_phase": args_cli.num_rollouts_per_phase,
            "overall_rewards": episode_rewards,
            "overall_lengths": episode_lengths,
            "per_phase_metrics": {str(phase): metrics for phase, metrics in phase_metrics.items()}
        }
        with open(raw_data_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        print(f"[INFO] Raw data saved to: {raw_data_file}")
    else:
        print("[WARNING] No episodes completed during evaluation")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()