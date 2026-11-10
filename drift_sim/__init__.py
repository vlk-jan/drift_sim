"""Differentiable skid-steer simulator over terrain-property maps (torch and Warp)."""

from drift_sim.simulator import Simulator
from drift_sim.simulator_wp import Simulator as WarpSimulator
from drift_sim.utils import (
    ROBOT_MODELS,
    PhysicsEngineConfig,
    RobotModelConfig,
    WorldConfig,
    robot_model_config,
)

__all__ = [
    "ROBOT_MODELS",
    "PhysicsEngineConfig",
    "RobotModelConfig",
    "Simulator",
    "WarpSimulator",
    "WorldConfig",
    "robot_model_config",
]
