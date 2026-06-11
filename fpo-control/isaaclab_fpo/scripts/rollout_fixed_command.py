#!/usr/bin/env python3
"""
Rollout script with fixed velocity command for comparing FPO and PPO gaits.

This script:
1. Runs a single robot from a known start state
2. Uses a fixed "move forward" velocity command
3. Collects robot states (position, orientation, joint positions/velocities) for N timesteps
4. Saves states to an npz file for later visualization

Example usage:
    # FPO rollout
    python rollout_fixed_command.py \
        --task Isaac-Velocity-Flat-Spot-v0 \
        --checkpoint ./logs/fpo_rsl_rl/spot_flat_flow/2025-11-04_23-58-35_20251104_test_test_test/model1499.pt \
        --output fpo_rollout.npz \
        --num_steps 200

    # PPO rollout
    python rollout_fixed_command.py \
        --task Isaac-Velocity-Flat-Spot-v0 \
        --checkpoint ./logs/rsl_rl/spot_flat/2025-11-05_00-04-25_20251104_test_test_test/model1499.pt \
        --output ppo_rollout.npz \
        --num_steps 200 \
        --use_ppo
"""

import viser  # HACK: import before isaaclab

import argparse
from pathlib import Path
import os
import sys

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Rollout with fixed command for gait comparison.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--output", type=str, required=True, help="Output npz file path.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of timesteps to rollout.")
parser.add_argument("--use_ppo", action="store_true", help="Use PPO runner instead of Flow runner.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--flow-sampling-steps", type=int, default=None, help="Number of sampling steps for flow matching (for FPO only).")

# Fixed command parameters
parser.add_argument("--lin_vel_x", type=float, default=1.0, help="Linear velocity x command (m/s).")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="Linear velocity y command (m/s).")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="Angular velocity z command (rad/s).")

# Network architecture parameters (to match checkpoint)
parser.add_argument("--actor_hidden_dims", type=str, default=None, help="Actor hidden dimensions (e.g., '[256,256,256]').")
parser.add_argument("--critic_hidden_dims", type=str, default=None, help="Critic hidden dimensions (e.g., '[256,256,256]').")
parser.add_argument("--empirical-normalization", action="store_true", help="Enable empirical observation normalization (required for checkpoints trained with normalization).")

# append FPO-RSL-RL or RSL-RL cli arguments (includes --checkpoint, --resume, --load_run, etc.)
cli_args.add_fpo_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force single environment
args_cli.num_envs = 1
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

if args_cli.use_ppo:
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
else:
    from fpo_rsl_rl.runners import OnPolicyRunner
    from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper, export_policy_as_jit

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_pickle

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Run rollout with fixed command."""
    task_name = args_cli.task.split(":")[-1]

    # Get checkpoint path
    resume_path = retrieve_file_path(args_cli.checkpoint)

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
        if args_cli.use_ppo:
            # Use RSL-RL config for PPO
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
        else:
            # Use FPO-RSL-RL config for FPO
            agent_cfg = cli_args.parse_fpo_rsl_rl_cfg(task_name, args_cli)

    # Override hidden dims if specified via command line
    if args_cli.actor_hidden_dims is not None:
        import ast
        actor_dims = ast.literal_eval(args_cli.actor_hidden_dims)
        print(f"[INFO] Overriding actor_hidden_dims from {agent_cfg.policy.actor_hidden_dims} to {actor_dims}")
        agent_cfg.policy.actor_hidden_dims = actor_dims
    if args_cli.critic_hidden_dims is not None:
        import ast
        critic_dims = ast.literal_eval(args_cli.critic_hidden_dims)
        print(f"[INFO] Overriding critic_hidden_dims from {agent_cfg.policy.critic_hidden_dims} to {critic_dims}")
        agent_cfg.policy.critic_hidden_dims = critic_dims

    # Override empirical normalization if specified
    if args_cli.empirical_normalization:
        print(f"[INFO] Enabling empirical normalization (was: {agent_cfg.empirical_normalization})")
        agent_cfg.empirical_normalization = True

    # Load environment config
    if os.path.exists(env_pkl_path):
        print(f"[INFO] Loading environment config from: {env_pkl_path}")
        env_cfg = load_pickle(env_pkl_path)
        # Override settings
        env_cfg.scene.num_envs = 1
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
    else:
        print(f"[WARNING] No saved environment config found at {env_pkl_path}, using default config")
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
        )

    print(f"[INFO] Loading experiment from directory: {log_dir}")
    print(f"[INFO] Using policy network hidden dims: {agent_cfg.policy.actor_hidden_dims}")

    # Override flow sampling steps if specified
    if not args_cli.use_ppo and args_cli.flow_sampling_steps is not None and hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Overriding flow sampling steps from {agent_cfg.policy.sampling_steps} to {args_cli.flow_sampling_steps}")
        agent_cfg.policy.sampling_steps = args_cli.flow_sampling_steps
    elif not args_cli.use_ppo and hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Using flow sampling steps: {agent_cfg.policy.sampling_steps}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Save reference to base environment
    base_env = env.unwrapped

    # wrap around environment
    if args_cli.use_ppo:
        env = RslRlVecEnvWrapper(env)
    else:
        env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Apply the sampling steps override to the loaded policy if it was changed
    if not args_cli.use_ppo and args_cli.flow_sampling_steps is not None and hasattr(ppo_runner.alg.policy, 'sampling_steps'):
        ppo_runner.alg.policy.sampling_steps = args_cli.flow_sampling_steps

    # obtain the trained policy for inference
    base_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # DEBUG: Check if normalizer is being used and wrap policy to log normalization
    policy_type = "PPO" if args_cli.use_ppo else "FPO"
    print(f"[DEBUG] {policy_type} empirical_normalization: {ppo_runner.empirical_normalization}")

    if ppo_runner.empirical_normalization:
        print(f"[DEBUG] {policy_type} obs_normalizer mean[9:12]: {ppo_runner.obs_normalizer._mean[0, 9:12]}")
        print(f"[DEBUG] {policy_type} obs_normalizer std[9:12]: {ppo_runner.obs_normalizer._std[0, 9:12]}")

        # Wrap the policy to log what the normalizer sees and outputs
        step_counter = [0]  # Use list to make it mutable in closure

        def logging_policy(obs):
            """Wrapper that logs normalization process"""
            if step_counter[0] < 3:  # Log first 3 steps
                print(f"\n[DEBUG] Step {step_counter[0]} - Before normalization:")
                print(f"  Raw obs[9:12] (vel cmd): {obs[0, 9:12]}")

                # Manually apply normalization to see the result
                normalized_obs = (obs - ppo_runner.obs_normalizer._mean) / (ppo_runner.obs_normalizer._std + ppo_runner.obs_normalizer.eps)
                print(f"  Normalized obs[9:12]: {normalized_obs[0, 9:12]}")

            step_counter[0] += 1
            return base_policy(obs)

        policy = logging_policy
    else:
        policy = base_policy

    # Reset environment to get initial state
    print(f"[INFO] Resetting environment...")
    obs, _ = env.reset()

    # Override the velocity command to be fixed
    fixed_command = torch.tensor([[args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z]],
                                  device=base_env.device, dtype=torch.float32)
    print(f"[INFO] Using fixed velocity command: [{args_cli.lin_vel_x}, {args_cli.lin_vel_y}, {args_cli.ang_vel_z}]")

    # Access the command manager and override the command
    if hasattr(base_env, 'command_manager') and 'base_velocity' in base_env.command_manager._terms:
        base_env.command_manager._terms['base_velocity'].command[:] = fixed_command
        print(f"[INFO] Successfully overrode command manager")

        # Get fresh observations with the new command
        obs, _ = env.get_observations()
        print(f"[INFO] Got fresh observations with fixed command")
    else:
        print(f"[WARNING] Could not find command manager, will try to override observations directly")

    # Storage for states
    root_pos_list = []
    root_quat_list = []
    joint_pos_list = []
    joint_vel_list = []
    base_lin_vel_list = []
    base_ang_vel_list = []
    actions_list = []
    rewards_list = []

    print(f"[INFO] Starting rollout for {args_cli.num_steps} steps...")

    # Run rollout
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            # Collect robot state BEFORE taking action (current state)
            robot = base_env.scene["robot"]
            root_pos_list.append(robot.data.root_pos_w.cpu().numpy())
            root_quat_list.append(robot.data.root_quat_w.cpu().numpy())
            joint_pos_list.append(robot.data.joint_pos.cpu().numpy())
            joint_vel_list.append(robot.data.joint_vel.cpu().numpy())
            base_lin_vel_list.append(robot.data.root_lin_vel_b.cpu().numpy())
            base_ang_vel_list.append(robot.data.root_ang_vel_b.cpu().numpy())

            # Get action from policy (logging happens inside the policy wrapper)
            actions = policy(obs)
            actions_list.append(actions.cpu().numpy())

            # Step environment
            obs, rewards, dones, _ = env.step(actions)
            rewards_list.append(rewards.cpu().numpy())

            # CRITICAL: Set command AFTER stepping, then get fresh observations
            # This ensures the next iteration sees the correct command
            if hasattr(base_env, 'command_manager') and 'base_velocity' in base_env.command_manager._terms:
                base_env.command_manager._terms['base_velocity'].command[:] = fixed_command
                # Get fresh observations with the fixed command
                obs, _ = env.get_observations()

            if step % 50 == 0:
                print(f"[INFO] Step {step}/{args_cli.num_steps}")

    # Convert lists to arrays
    states = {
        'root_pos': np.array(root_pos_list),  # Shape: (num_steps, 1, 3)
        'root_quat': np.array(root_quat_list),  # Shape: (num_steps, 1, 4)
        'joint_pos': np.array(joint_pos_list),  # Shape: (num_steps, 1, 12)
        'joint_vel': np.array(joint_vel_list),  # Shape: (num_steps, 1, 12)
        'base_lin_vel': np.array(base_lin_vel_list),  # Shape: (num_steps, 1, 3)
        'base_ang_vel': np.array(base_ang_vel_list),  # Shape: (num_steps, 1, 3)
        'actions': np.array(actions_list),  # Shape: (num_steps, 1, 12)
        'rewards': np.array(rewards_list),  # Shape: (num_steps, 1)
        'command': fixed_command.cpu().numpy(),  # Shape: (1, 3)
    }

    # Save to npz file
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **states)
    print(f"[INFO] Saved rollout states to: {output_path}")
    print(f"[INFO] Total reward: {states['rewards'].sum():.2f}")
    print(f"[INFO] Mean reward per step: {states['rewards'].mean():.4f}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
