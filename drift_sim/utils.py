from dataclasses import dataclass

import torch
from tensordict import TensorClass

from simulator.numerical import unit_quaternion


@dataclass
class RobotModelConfig:
    mass: float = 0
    torque_min: float = -1e9
    torque_max: float = -1e9
    # vector from the center of the triangle connecting the wheel centers to COG
    cog: torch.Tensor = torch.tensor([0, 0, 0])
    wheel_radii: torch.Tensor = torch.tensor([1, 1, 1])
    wheel_centers: torch.Tensor = torch.tensor(((1, -1, 0), (1, 1, 0), (-1, 0, 0)))
    wheel_inertias: torch.Tensor = torch.tensor(())  # TODO:
    inertia_tensor: torch.Tensor = torch.eye(3)


@dataclass
class PhysicsEngineConfig:
    """
    Physics Engine configuration. Contains the physical constants of the simulation, such as gravity, time step, etc.

    Attributes:
        dt (float): time step for numerical integration. Default is 0.01. (100 Hz)
        gravity (float): acceleration due to gravity. Default is 9.81 m/s^2.
        torque_limit (float): torque limit that can be generated on CoG. The physics engine clips it to this value. Default is 200.0 Nm.
        damping_alpha (float): damping coefficient modifier, should be between larger than 1.0. Applies damping alpha times higher
        than critical damping.
        soft_contact_sigma (float): soft contact sigma. Default is 0.5. This is the width of the soft contact force function.
    """

    num_robots: int = 1
    dt: float = 1e-2
    g: float = 9.81
    torque_limit: float = 200.0  # TODO:
    damping_alpha: float = 1.0
    soft_contact_sigma: float = 0.01  # TODO:


class PhysicsState(TensorClass):
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (B, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (B, 3).
        q (torch.Tensor): Orientation quaternion of the robot in the world frame. Shape (B, 4).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (B, 3).
        wheel_omega (torch.Tensor): Angular velocities of individual wheels. Shape (B, 3).
    """

    x: torch.Tensor
    xd: torch.Tensor
    q: torch.Tensor
    omega: torch.Tensor
    wheel_omega: torch.Tensor

    @staticmethod
    def dummy(**kwargs) -> "PhysicsState":
        """Create an empty dummy PhysicsState object with zero tensors.
        Some fields can be overridden by passing them as keyword arguments.
        """
        batch_size = kwargs.pop("batch_size", None)
        device = kwargs.pop("device", torch.get_default_device())
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError(
                    "Specified batch size does not match the shape of the tensors."
                )
            batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = {
            "x": torch.zeros(batch_size, 3),
            "xd": torch.zeros(batch_size, 3),
            "q": unit_quaternion(batch_size, device=device),
            "omega": torch.zeros(batch_size, 3),
            "wheel_omega": torch.zeros(batch_size, 3, device=device),
            "batch_size": batch_size,
        }
        return PhysicsState(**{**base, **kwargs}, device=device)


class PhysicsStateDer(TensorClass):
    """Physics State Derivative

    Attributes:
        xdd (torch.tensor): Acceleration of the robot in the world frame. Shape (B, 3).
        omega_d (torch.tensor): Derivative of the angular velocity of the robot in the world frame. Shape (B, 3).
        wheel_omega_d (torch.Tensor): Angular acceleration of the 3 wheels. Shape (B, 3).
    """

    xdd: torch.Tensor
    omega_d: torch.Tensor
    wheel_omega_d: torch.Tensor
