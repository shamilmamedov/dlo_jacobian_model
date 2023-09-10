import numpy as np
import torch

from RBF import JacobianPredictor 

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
        data = self._load_simulation_trajectory(n_traj)

        self.goal_pos = torch.tensor(data[[1], 87:117])
        self.fps_pos = torch.tensor(data[[1], 1:31])
        self.fps_vel = torch.tensor(data[[1], 45:75])
        self.ees_pose = torch.tensor(data[[1], 31:45])

    def _load_simulation_trajectory(self, n_traj):
        data_dir = '../../dataset/sim_data/'
        traj_name = f'state_{n_traj}.npy'
        filepath = data_dir + traj_name
        
        data = np.load(filepath)
        return data
    
    def step(self, action):
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        self.ees_pose = self.jp.calcNextEndsPose(
            self.ees_pose, action, self.dt
        )

        self.fps_pos = self.jp.predNextFPsPositions(
            self.length, self.fps_pos, self.ees_pose, action, self.dt
        )

    def _append_to_history(self):
        self.fps_pos_history.append(self.fps_pos)
        self.ees_pose_history.append(self.ees_pose)

    def history(self):
        return self.fps_pos_history, self.ees_pose_history


if __name__ == '__main__':
    env = DLOEnv()
    env.reset()
    env.step(np.zeros(12))
    print(env.ees_pose)
    print(env.fps_pos)


