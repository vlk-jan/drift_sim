import torch

from models.simulator import Simulator as TorchSimulator
from simulator.engine_wp import WarpStaticEngine
from simulator.utils import num_sim_steps
from utils.geometry import quat_to_rotmat


class Simulator(TorchSimulator):
    def __init__(
        self,
        robot="helhest_v2_median",
        device=None,
        T_horizon=1.0,
        dt=2e-3,
        com_transfer=True,
    ):
        """Warp simulator wrapper; T_horizon must match models/simulator.py.

        predict_trajectory runs four kernel launches per physics step, replayed
        from captured graphs; a rollout that needs gradients records a Warp tape
        (see simulator/engine_wp.py). vel_ts are absolute Unix timestamps,
        (B, Steps), and each element has its own window.
        """
        super().__init__(robot, device, T_horizon, dt, com_transfer)
        self._geom = self.engine
        del self.engine
        self.engine = WarpStaticEngine(self.robot_model, self.device)
        self.engine_config = self._geom.config

    def _prepare_inputs(self, vel_ctrl, vel_ts, T_horizon=None):
        sim_dt = self.engine_config.dt
        horizon = self.T_horizon if T_horizon is None else T_horizon
        num_steps = num_sim_steps(horizon, sim_dt)

        # Build the query timeline per element: windows are far apart in Unix time,
        # and losses match poses by nearest timestamp, so a shared timeline would
        # silently zero the gradient of every element but the first.
        B, S = vel_ts.shape
        # float64 is mandatory: float32 spacing at ~1.76e9 s is ~128 s.
        vel_ts = vel_ts.to(self.device, torch.float64).contiguous()
        vel_ctrl = vel_ctrl.to(self.device)
        step_offsets = (
            torch.arange(num_steps, device=vel_ts.device, dtype=torch.float64) * sim_dt
        )
        # Query at each sim step's start -- the control that drives that step.
        query_times = vel_ts[:, :1] + step_offsets  # (B, num_steps)

        idx = torch.searchsorted(vel_ts, query_times, right=True) - 1
        idx = idx.clamp(0, S - 2)
        batch_idx = torch.arange(B, device=vel_ts.device).unsqueeze(1)

        t0 = vel_ts[batch_idx, idx]
        t1 = vel_ts[batch_idx, idx + 1]
        v0 = vel_ctrl[batch_idx, idx]  # (B, num_steps, 3)
        v1 = vel_ctrl[batch_idx, idx + 1]

        alpha = ((query_times - t0) / (t1 - t0)).unsqueeze(-1)
        # Clamped like the torch engine (hold, don't invent commands); collate pads
        # ctrl_ts by repeating the last stamp, so padded 0/0 hits nan_to_num.
        alpha = torch.nan_to_num(alpha, 0.0).clamp(0.0, 1.0)

        # Dense velocities, (SimSteps, B, 3).
        dense_controls = (v0 + (v1 - v0) * alpha).permute(1, 0, 2).to(torch.float32)
        return dense_controls, query_times

    def _rollout(self, state, dense_ctrl, query_times, world_config, include_start):
        traj_x, traj_q, traj_xd, traj_omega = self.engine.run_trajectory(
            state, dense_ctrl, world_config, self.engine_config
        )
        if include_start:
            traj_x, traj_q, traj_xd, traj_omega = (
                torch.cat([s.unsqueeze(0), t])
                for s, t in zip(
                    (state.x, state.q, state.xd, state.omega),
                    (traj_x, traj_q, traj_xd, traj_omega),
                )
            )

        # engine_wp returns wxyz quaternions.
        R = quat_to_rotmat(traj_q)  # (SimSteps, B, 3, 3)
        x = traj_x  # (SimSteps, B, 3)

        T = (
            torch.eye(4, device=self.device)
            .view(1, 1, 4, 4)
            .repeat(x.shape[0], state.x.shape[0], 1, 1)
        )
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = x

        T = T.permute(1, 0, 2, 3)  # (B, SimSteps, 4, 4)
        xd_out = traj_xd.permute(1, 0, 2)  # (B, SimSteps, 3)
        omega_out = traj_omega.permute(1, 0, 2)  # (B, SimSteps, 3)

        # Poses are post-step states stamped at the step's end: engine_wp drops the
        # seeded step 0, so its pose i is the state after step i+1; include_start
        # puts the start, stamped vel_ts[:, 0], in front.
        timestamps = query_times + self.engine_config.dt  # (B, SimSteps), float64
        if include_start:
            timestamps = torch.cat([query_times[:, :1], timestamps], dim=1)
        return T, timestamps, xd_out, omega_out
