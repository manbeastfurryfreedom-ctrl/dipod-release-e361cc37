"""IsaacLab extension for Flow Policy Optimization (FPO).

This package provides the IsaacLab integration layer for fpo_rsl_rl:
- Config classes (runner, actor-critic, algorithm)
- VecEnv wrapper for IsaacLab environments
- Policy exporters (JIT, ONNX)
- Per-task training configs with a simple registry
"""


def __getattr__(name):
    if name in ("FpoRslRlOnPolicyRunnerCfg", "FpoRslRlPpoActorCriticCfg", "FpoRslRlPpoAlgorithmCfg", "FpoSelfDistillationCfg"):
        from .rl_cfg import FpoRslRlOnPolicyRunnerCfg, FpoRslRlPpoActorCriticCfg, FpoRslRlPpoAlgorithmCfg, FpoSelfDistillationCfg
        return locals()[name]
    if name == "FpoRslRlVecEnvWrapper":
        from .wrapper import FpoRslRlVecEnvWrapper
        return FpoRslRlVecEnvWrapper
    if name in ("export_policy_as_jit", "export_policy_as_onnx"):
        from .exporter import export_policy_as_jit, export_policy_as_onnx
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
