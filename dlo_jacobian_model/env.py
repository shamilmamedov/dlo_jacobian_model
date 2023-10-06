import torch

from RBF import JacobianPredictor
from dlo_jacobian_model.utils.data import load_simulation_trajectory


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
        self.action_history = []

    def reset(self, n_traj: int = 0):
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