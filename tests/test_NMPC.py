import numpy as np
import torch
import pytest


from RBF import JacobianPredictor 
import dlo_jacobian_model.nonlinear_MPC as nonlinear_MPC
import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model
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

    def reset(self):
        n_traj = 0
        data = load_simulation_trajectory(n_traj)

        self.goal_pos = torch.tensor(data[[1], 87:117])
        self.fps_pos = torch.tensor(data[[1], 1:31])
        self.fps_vel = torch.tensor(data[[1], 45:75])
        self.ees_pose = torch.tensor(data[[1], 31:45])

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




class TestNMPC:
    nmpcs_options = nonlinear_MPC.NMPCOptions(
        dt=0.1,
        N=10,
        u_max=np.ones((12,)),
        u_min=-np.ones((12,)),
    )
    
    def _create_model(self):
        dlo_model_parms = casadi_dlo_model.load_model_parameters()
        dlo_model = casadi_dlo_model.JacobianNetwork(**dlo_model_parms)
        dlo_length = 0.5
        setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)
        return setup_model

    def test_instantiation(self):
        setup_model = self._create_model()
        nmpc = nonlinear_MPC.NMPC(setup_model, self.nmpcs_options)

        assert nmpc.iter_counter == 0
        assert nmpc.options.dt == 0.1
        assert nmpc.options.N == 10
        assert nmpc.options.tf == 1.0

    def test_solve(self):
        n_traj = 0
        data = load_simulation_trajectory(n_traj)
        fps_pos = torch.tensor(data[[1], 1:31]).numpy().ravel()
        ees_pose = torch.tensor(data[[1], 31:45]).numpy().ravel()
        goal_pos = torch.tensor(data[[1], 87:117]).numpy().ravel()

        z = np.concatenate((fps_pos, ees_pose))
        z_ref = np.hstack((goal_pos, ees_pose)).T

        setup_model = self._create_model()
        # self.nmpcs_options.build_ocp_solver = False
        nmpc = nonlinear_MPC.NMPC(setup_model, self.nmpcs_options)
        nmpc.set_reference(goal_pos, ees_pose)
        nmpc(z)   
        breakpoint()
        print('AAAA')



if __name__ == '__main__':
    # env = DLOEnv()
    # env.reset()
    # env.step(np.zeros(12))
    # print(env.ees_pose)
    # print(env.fps_pos)
    # test_instantiation()

    t = TestNMPC()
    t.test_solve()
