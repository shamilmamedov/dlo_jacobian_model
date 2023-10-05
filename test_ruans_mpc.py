import casadi as cs
import matplotlib.pyplot as plt
import torch
import copy
import numpy as np

from utils.data import load_simulation_trajectory
import casadi_dlo_model
from ruans_MPC import RuansMPC, RuansMPCOptions
from RBF import JacobianPredictor
import visualization


class DLOEnv:
    def __init__(self, dt: float = 0.1, length: float = 0.5) -> None:
        self.dt = torch.tensor(dt, dtype=torch.float32)
        self.length = torch.tensor(length, dtype=torch.float32)

        self.jp = JacobianPredictor()
        self.jp.LoadModelWeights()

        self.goal_pos = None
        self.fps_pos = None
        self.ees_pose = None

        self.fps_pos_history = []
        self.ees_pose_history = []

    def reset(self):
        n_traj = 0
        data = load_simulation_trajectory(n_traj)

        self.goal_pos = torch.tensor(data[[1], 87:117])
        self.fps_pos = torch.tensor(data[[1], 1:31])
        self.fps_vel = torch.tensor(data[[1], 45:75])
        self.ees_pose = torch.tensor(data[[1], 31:45])

        z = torch.hstack((self.fps_pos, self.ees_pose))
        z = z.squeeze(0).numpy()
        return z, self.goal_pos.squeeze(0).numpy()

    def step(self, action):
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        self.ees_pose = torch.tensor(
            self.jp.calcNextEndsPose(
                self.ees_pose, action, self.dt
            )
        )

        self.fps_pos = self.jp.predNextFPsPositions(
            self.length, self.fps_pos, self.ees_pose, action, self.dt
        )
        self._append_to_history()

        z = torch.hstack((self.fps_pos, self.ees_pose))
        z = z.squeeze(0).numpy()
        return z

    def _append_to_history(self):
        self.fps_pos_history.append(self.fps_pos)
        self.ees_pose_history.append(self.ees_pose)

    def history(self):
        return self.fps_pos_history, self.ees_pose_history


class TestLinearizedMPC:
    lmpc_opts = RuansMPCOptions(
        dt=0.1,
        N=10,
        u_max=np.ones((12,)),
        u_min=-np.ones((12,)),
    )

    def _create_model(self):
        dlo_model_parms = casadi_dlo_model.load_model_parameters()
        dlo_model = casadi_dlo_model.JacobianNetwork(**dlo_model_parms)
        return dlo_model

    def test_closed_loop(self):
        model = self._create_model()
        rmpc = RuansMPC(model, self.lmpc_opts)

        env = DLOEnv()
        z, fps_pos_des = env.reset()
        for k in range(40):
            u = rmpc(z, fps_pos_des)
            z = env.step(u)

        fps_pos = np.concatenate(env.fps_pos_history, axis=0)
        fps_pos_des = np.tile(fps_pos_des, (fps_pos.shape[0], 1))

        visualization.visualize_mass_spring_system(
            fps_pos, fps_pos_des, 0.1, 5
        )


if __name__ == '__main__':
    t = TestLinearizedMPC()
    t.test_closed_loop()