"""Isaac Lab task: a UR5 swings a GPU Cosserat rope (ropesim) so its tip passes through a goal point along a goal direction."""
import gymnasium as gym

gym.register(
    id="RopeSwing-v0",
    entry_point=f"{__name__}.swing_env:RopeSwingEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.swing_env:RopeSwingEnvCfg"},
)
