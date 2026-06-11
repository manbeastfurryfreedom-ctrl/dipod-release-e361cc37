"""Self-distillation pre-training for FPO flow policies.

Freezes a copy of the randomly initialized teacher policy, generates
(observation, action) pairs from the teacher on live environment rollouts,
and trains only the student actor/flow model via the existing CFM loss
so the sampling path better matches the forward/backward flow process.

The critic is not trained during self-distillation.
"""

from __future__ import annotations

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim


def run_self_distillation(
    *,
    policy: nn.Module,
    env,
    obs_normalizer: nn.Module,
    device: str,
    num_iterations: int = 100,
    rollout_steps: int = 8,
    batch_size: int = 16384,
    learning_rate: float = 3e-4,
    n_samples_per_action: int = 16,
    writer=None,
    logger_type: str = "tensorboard",
) -> dict:
    """Run self-distillation pre-training phase.

    Args:
        policy: The ActorCritic policy (student). Only actor params are updated.
        env: The VecEnv wrapper around the Isaac Lab environment.
        obs_normalizer: EmpiricalNormalization or Identity module for observations.
        device: Torch device string.
        num_iterations: Number of distillation gradient steps.
        rollout_steps: Environment steps per data collection round.
        batch_size: Mini-batch size for each gradient step.
        learning_rate: Learning rate for distillation optimizer.
        n_samples_per_action: Samples per action for CFM loss (matches FPO default).
        writer: Optional tensorboard/wandb writer for logging.
        logger_type: Type of logger ("tensorboard", "wandb", etc.).

    Returns:
        Dict with summary metrics: loss history, wall time, sample count.
    """
    t_start = time.perf_counter()

    teacher_actor = copy.deepcopy(policy.actor)
    for p in teacher_actor.parameters():
        p.requires_grad = False
    teacher_actor.eval()

    distill_optimizer = optim.AdamW(
        policy.actor.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    obs_buffer = []
    action_buffer = []

    loss_history = []
    action_mse_history = []
    total_samples = 0

    was_training = policy.training
    policy.train()

    if hasattr(obs_normalizer, "train"):
        obs_normalizer.train()

    obs, extras = env.get_observations()
    obs = obs.to(device)

    for it in range(num_iterations):
        with torch.no_grad():
            for _ in range(rollout_steps):
                norm_obs = obs_normalizer(obs)
                teacher_actions = _teacher_act(
                    policy, teacher_actor, norm_obs, device
                )
                obs_buffer.append(norm_obs)
                action_buffer.append(teacher_actions)

                obs, _rew, _dones, _infos = env.step(teacher_actions.to(env.device))
                obs = obs.to(device)

        all_obs = torch.cat(obs_buffer, dim=0)
        all_actions = torch.cat(action_buffer, dim=0)
        total_samples += all_obs.shape[0]

        n_available = all_obs.shape[0]
        effective_batch = min(batch_size, n_available)
        indices = torch.randperm(n_available, device=device)[:effective_batch]
        batch_obs = all_obs[indices]
        batch_actions = all_actions[indices]

        num_act = policy.num_actions
        eps = torch.randn(
            (effective_batch, n_samples_per_action, num_act), device=device
        )

        beta = policy.cfm_loss_t_inverse_cdf_beta
        uniform_t = torch.rand(
            (effective_batch, n_samples_per_action, 1), device=device
        )
        t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))

        loss_per_sample, _x1, _x0 = policy.get_cfm_loss(
            batch_obs, batch_actions, eps, t
        )
        loss = loss_per_sample.mean()

        distill_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
        distill_optimizer.step()

        with torch.no_grad():
            student_actions = policy.act(batch_obs)
            action_mse = ((student_actions - batch_actions) ** 2).mean().item()

        loss_val = loss.item()
        loss_history.append(loss_val)
        action_mse_history.append(action_mse)

        if writer is not None:
            writer.add_scalar("SelfDistill/loss", loss_val, it)
            writer.add_scalar("SelfDistill/action_mse_to_teacher", action_mse, it)

        if it % max(1, num_iterations // 10) == 0 or it == num_iterations - 1:
            print(
                f"  [SelfDistill] iter {it}/{num_iterations}  "
                f"loss={loss_val:.6f}  action_mse={action_mse:.6f}"
            )

        max_buffer_size = max(batch_size * 4, env.num_envs * rollout_steps * 8)
        if n_available > max_buffer_size:
            obs_buffer = obs_buffer[len(obs_buffer) // 2 :]
            action_buffer = action_buffer[len(action_buffer) // 2 :]

    wall_time = time.perf_counter() - t_start

    if not was_training:
        policy.eval()

    del teacher_actor
    del distill_optimizer
    torch.cuda.empty_cache()

    summary = {
        "wall_time_sec": wall_time,
        "num_iterations": num_iterations,
        "total_samples": total_samples,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "final_action_mse": action_mse_history[-1] if action_mse_history else 0.0,
        "loss_history": loss_history,
        "action_mse_history": action_mse_history,
    }

    print(
        f"  [SelfDistill] Done: {num_iterations} iters, "
        f"{wall_time:.1f}s wall time, {total_samples} samples, "
        f"final_loss={summary['final_loss']:.6f}"
    )

    if writer is not None:
        writer.add_scalar("SelfDistill/wall_time_sec", wall_time, 0)
        writer.add_scalar("SelfDistill/total_samples", total_samples, 0)
        writer.add_scalar("SelfDistill/iterations", num_iterations, 0)

    return summary


def _teacher_act(
    policy: nn.Module,
    teacher_actor: nn.Module,
    observations: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Generate actions using the frozen teacher actor with the student's flow integration."""
    batch_size = observations.shape[0]
    num_actions = policy.num_actions

    x_t = torch.randn(size=(batch_size, num_actions), device=device)

    sampling_steps = policy.sampling_steps
    t_path = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)
    t_current = t_path[:-1]
    dt = t_path[1:] - t_path[:-1]

    half_dim = policy.timestep_embed_dim // 2
    freqs = 2 ** torch.arange(half_dim, device=device, dtype=observations.dtype)

    for i in range(sampling_steps):
        t_val = t_current[i].reshape(1, 1)
        scaled_t = t_val * freqs
        embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
        embedded_t = embedded_t.expand(batch_size, -1)

        mlp_output = teacher_actor(
            torch.cat([observations, embedded_t, x_t], dim=-1)
        )
        mlp_output = policy.mlp_output_scale * mlp_output
        u = mlp_output
        x_t = x_t + u * dt[i]

    actions = policy.actor_scale * x_t
    return actions
