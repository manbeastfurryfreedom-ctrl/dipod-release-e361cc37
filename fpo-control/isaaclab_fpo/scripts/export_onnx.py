#!/usr/bin/env python3

"""Launch Isaac Sim Simulator first."""
import viser
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export a trained FPO++ policy to ONNX format.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--file-name", type=str, default="fpo_policy.onnx", help="Name of the exported ONNX file.")

# Viser-specific arguments
parser.add_argument("--viser", action="store_true", default=False, help="Enable Viser visualization.")
parser.add_argument("--viser-port", type=int, default=8080, help="Port for Viser web server.")
parser.add_argument("--asset-dir", type=str, default=None, help="Directory containing pre-extracted assets.")
parser.add_argument("--viser-update-freq", type=int, default=1, help="Update Viser every N steps.")
parser.add_argument("--viser-env-spacing", type=float, default=1.5, help="Spacing between environments in regular grid visualization (default: 1.5m).")
parser.add_argument("--viser-fps", type=int, default=60, help="Target frame rate for Viser visualization (default: 60).")
parser.add_argument("--viser-random-grid-size", type=float, default=0.0, help="Size of grid for random robot offsets. Set to 0.0 to disable random offsets (default: 3x3).")

# Flow matching specific arguments
parser.add_argument("--flow-sampling-steps", type=int, default=None, help="Number of sampling steps for flow matching inference (default: use training value).")
parser.add_argument("--training-sampling-steps", type=int, default=None, help="Number of sampling steps for training (default: use training value).")
# Motion tracking specific arguments
parser.add_argument("--testing_phase", type=float, default=None, help="Testing phase for motion tracking (e.g., --testing_phase 0.141). If provided, environment will randomly sample from these phases.")
parser.add_argument("--zero_noise_sampling", action="store_true", default=False, help="Use zero noise sampling instead of random noise for flow model (overrides config)")

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
import sys
import time
import torch
import copy
import wandb
from pathlib import Path

from fpo_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_pickle
from isaaclab_fpo.viser import ViserIsaacLab, ViserMotionTracking

from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



import onnx
import onnxruntime as ort
import torch
from isaaclab_fpo.exporter import _OnnxPolicyExporter
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

from isaaclab.envs import ManagerBasedRLEnv


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


################## Export helping functions and classes ################## 
def export_motion_policy_as_onnx(
        env: ManagerBasedRLEnv, actor_critic: object, path: str, normalizer: object | None = None, testing_phase: float | None = None,
        filename="policy.onnx",
        verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose, testing_phase)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):

    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, testing_phase: float | None = None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.policy = actor_critic

        self.joint_pos = cmd.motion.joint_pos.to("cpu") 
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

        if testing_phase is not None:
            starting_idx = int(self.time_step_total * testing_phase)
            self.joint_pos = self.joint_pos[starting_idx:]
            self.joint_vel = self.joint_vel[starting_idx:]
            self.body_pos_w = self.body_pos_w[starting_idx:]
            self.body_quat_w = self.body_quat_w[starting_idx:]
            self.body_lin_vel_w = self.body_lin_vel_w[starting_idx:]
            self.body_ang_vel_w = self.body_ang_vel_w[starting_idx:]
            self.time_step_total = self.joint_pos.shape[0]
    
    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (self.policy.act_inference(self.normalizer(x)),
                self.joint_pos[time_step_clamped],
                self.joint_vel[time_step_clamped],
                self.body_pos_w[time_step_clamped],
                self.body_quat_w[time_step_clamped],
                self.body_lin_vel_w[time_step_clamped],
                self.body_ang_vel_w[time_step_clamped])

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.policy.num_actor_obs)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=17,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                          "body_ang_vel_w"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x)  # numbers → format, strings → as-is
        for x in arr
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    metadata = {"run_path": run_path,
                "joint_names": env.scene["robot"].data.joint_names,
                "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
                "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
                "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
                "command_names": env.command_manager.active_terms,
                "observation_names": env.observation_manager.active_terms["policy"],
                "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist()}

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)

class DynamicBatchONNXPolicy:
    """Wrapper to handle dynamic batching for fixed-batch ONNX model."""

    def __init__(self, onnx_path="policy.onnx"):
        """Initialize the ONNX policy wrapper.

        Args:
            onnx_path: Path to the ONNX model file (relative to exported/ directory)
        """
        import os
        if not os.path.isabs(onnx_path):
            # If relative path, assume it's in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            onnx_path = os.path.join(script_dir, onnx_path)

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = self.input_shape[1]

        print(f"Loaded ONNX model from: {onnx_path}")
        print(f"Input name: {self.input_name}, shape: {self.input_shape}")
        print(f"Observation dimension: {self.obs_dim}")

    def __call__(self, observations, time_step):
        """
        Run inference on a batch of observations.

        Args:
            observations: numpy array or torch tensor of shape (batch_size, obs_dim)

        Returns:
            actions: numpy array of shape (batch_size, action_dim)
        """

        batch_size = observations.shape[0]

        if batch_size == 1:
            # Direct inference for single batch (model expects batch size 1)
            return self.session.run(None, {self.input_name: observations, self.session.get_inputs()[1].name: time_step})
        else:
            # Process each sample individually since model has fixed batch size 1
            outputs = []
            for i in range(batch_size):
                obs_single = observations[i:i+1]  # Keep batch dimension
                output = self.session.run(None, {self.input_name: obs_single})[0]
                outputs.append(output)
            return torch.cat(outputs, dim=0)
            # return np.concatenate(outputs, axis=0)

    def act_inference(self, observations):
        """Alias for __call__ to match PyTorch interface."""
        return self.__call__(observations)

################## Main function ################## 
def main():
    """Export FPO-RSL-RL policy to ONNX format."""

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
        print(f"[INFO] Loaded config has num_envs: {env_cfg.scene.num_envs}. Overriding to 1.")
        # Override some settings for playback
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
    
    # Override testing phases for motion tracking if specified
    if args_cli.testing_phase is not None:
        # Check if this is a motion tracking task and set the testing phases
        if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
            if type(env_cfg.commands.motion).__name__ == 'MotionCommandCfg':
                print(f"[INFO] Setting motion tracking testing phases to: {args_cli.testing_phase}")
                env_cfg.commands.motion.testing_phases = [args_cli.testing_phase]
    
    # Override zero_noise_sampling if provided via command line
    if args_cli.zero_noise_sampling and hasattr(agent_cfg.policy, 'zero_noise_sampling'):
        agent_cfg.policy.zero_noise_sampling = True
        print(f"[INFO] Overriding config: zero_noise_sampling = True")

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

    # Save reference to base environment for Viser
    base_env = env.unwrapped

    # wrap around environment for fpo-rsl-rl
    env = FpoRslRlVecEnvWrapper(env, clip_actions=None)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    # extract the neural network module
    policy_nn = ppo_runner.alg.policy

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(env.unwrapped, policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, testing_phase=args_cli.testing_phase, filename=args_cli.file_name)
    attach_onnx_metadata(env.unwrapped, "none", export_model_dir, filename=args_cli.file_name)
    print(f"[INFO] Exported policy to ONNX format: {export_model_dir}/{args_cli.file_name}")
    
    ################## Initialize Viser visualization ################## 
    # Initialize Viser visualization
    viser_viz = None
    if args_cli.viser:
        print("\n[INFO] Initializing Viser visualization...")
        
        # Determine asset directory
        if args_cli.asset_dir:
            asset_dir = Path(args_cli.asset_dir)
        else:
            # Try to auto-detect based on task name
            task_clean = args_cli.task.lower().replace(":", "_").replace("-", "_")
            asset_dir = Path(__file__).resolve().parents[1] / "viser_assets" / task_clean
        
        if not asset_dir.exists():
            print(f"[ERROR] Asset directory not found: {asset_dir}")
            print("Please run isaac_asset_extractor.py first to extract the assets.")
            return
        
        try:
            # Get actual number of environments
            # You can adjust this cap based on your system's performance
            num_envs_to_viz = 1
            # Generate random offsets if grid size > 0
            random_offsets = None
            if args_cli.viser_random_grid_size > 0:
                # Generate random positions within the grid using single call
                half_size = args_cli.viser_random_grid_size / 2
                random_offsets = np.random.uniform(
                    low=[-half_size, -half_size, 0],
                    high=[half_size, half_size, 0],
                    size=(num_envs_to_viz, 3)
                )
                print(f"[INFO] Generated random offsets within {args_cli.viser_random_grid_size}x{args_cli.viser_random_grid_size} grid")
                # Pass random offsets to ViserIsaacLab
                env_spacing = 0.0  # No regular grid spacing when using random offsets
            else:
                env_spacing = args_cli.viser_env_spacing
            
            # Check if this is a motion tracking task
            is_motion_tracking = False
            if hasattr(base_env, 'command_manager'):
                for cmd_name, cmd_term in base_env.command_manager._terms.items():
                    if type(cmd_term).__name__ == 'MotionCommand':
                        is_motion_tracking = True
                        print(f"[INFO] Detected motion tracking task - will use ghost robot visualization")
                        break
            
            # Create appropriate visualizer
            if is_motion_tracking:
                viser_viz = ViserMotionTracking(
                    asset_dir=asset_dir,
                    port=args_cli.viser_port,
                    update_freq=args_cli.viser_update_freq,
                    num_envs=num_envs_to_viz,
                    env_spacing=env_spacing,
                    fps=args_cli.viser_fps,
                    random_offsets=random_offsets,
                    ghost_opacity=0.4,
                    ghost_color=(0.3, 1.0, 0.3),  # Green tint for reference
                )
            else:
                viser_viz = ViserIsaacLab(
                    asset_dir=asset_dir,
                    port=args_cli.viser_port,
                    update_freq=args_cli.viser_update_freq,
                    num_envs=num_envs_to_viz,
                    env_spacing=env_spacing,
                    fps=args_cli.viser_fps,
                    random_offsets=random_offsets,
                )
            
            # Load mapping from base environment
            viser_viz.load_from_env(base_env)
            
            print(f"[INFO] Viser server running at http://localhost:{args_cli.viser_port}")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize Viser: {e}")
            import traceback
            traceback.print_exc()
            viser_viz = None
            exit()

    # Load the exported onnx policy
    file_path = os.path.join(os.path.abspath(export_model_dir), args_cli.file_name)
    policy = DynamicBatchONNXPolicy(file_path)
    print(f"[INFO] Loaded ONNX policy from: {file_path}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    
    # Performance tracking for Viser
    viser_update_time = 0.0
    viser_update_count = 0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # Handle reset request with proper mode management
        reset_requested = viser_viz is not None and viser_viz.check_reset_request()
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            s1 = time.perf_counter()
            motion = env.unwrapped.command_manager.get_term('motion')
            ts = motion.time_steps
            actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = policy(obs.cpu().numpy(), ts.cpu().unsqueeze(0).numpy().astype(np.float32))
            actions = torch.from_numpy(actions).cuda()
            joint_pos = torch.from_numpy(joint_pos).cuda()
            joint_vel = torch.from_numpy(joint_vel).cuda()
            body_pos_w = torch.from_numpy(body_pos_w).cuda()
            body_quat_w = torch.from_numpy(body_quat_w).cuda()
            body_lin_vel_w = torch.from_numpy(body_lin_vel_w).cuda()
            body_ang_vel_w = torch.from_numpy(body_ang_vel_w).cuda()

            s2 = time.perf_counter()
            print(f"[INFO] Time taken for policy inference: {(s2 - s1) * 1000:.2f} ms")
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        
            if reset_requested:
                print(f"[INFO] Resetting environment...")
                # Use the wrapper's reset method which returns the proper observation format
                obs, _ = env.reset()
                print(f"[INFO] Reset complete")
        
        # Update Viser visualization
        if viser_viz is not None:
            
            viser_start = time.time()
            try:
                # Use the saved base environment reference and pass rewards and actions
                viser_viz.update_from_env(base_env, rewards=rewards, actions=actions)
            except Exception as e:
                print(f"[WARNING] Viser update failed: {e}")
            viser_update_time += time.time() - viser_start
            viser_update_count += 1
            
            # Print performance stats every 1000 steps
            if viser_update_count % 1000 == 0:
                avg_viser_time = viser_update_time / viser_update_count * 1000  # ms
                print(f"[VISER] Average update time: {avg_viser_time:.2f} ms")
        
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    # Close Viser
    if viser_viz is not None:
        viser_viz.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
