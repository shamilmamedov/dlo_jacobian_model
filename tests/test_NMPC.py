import numpy as np
import torch


import dlo_jacobian_model.nonlinear_MPC as nonlinear_MPC
import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model
from dlo_jacobian_model.utils.data import load_simulation_trajectory
import dlo_jacobian_model.visualization as visualization
from dlo_jacobian_model.env import DLOEnv


class TestNMPCAcados:
    nmpcs_options = nonlinear_MPC.NMPCOptions(
        dt=0.1,
        N=5,
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
        nmpc = nonlinear_MPC.NMPCAcados(setup_model, self.nmpcs_options)

        assert nmpc.iter_counter == 0
        assert nmpc.options.dt == 0.1
        assert nmpc.options.N == 10
        assert nmpc.options.tf == 1.0

    def test_solve(self):
        n_traj = 0
        data = load_simulation_trajectory(n_traj)
        fps_pos = data[1, 1:31]
        ees_pose = data[1, 31:45]
        goal_pos = data[1, 87:117]

        z = np.concatenate((fps_pos, ees_pose))
        z_ref = np.hstack((goal_pos, ees_pose)).T

        setup_model = self._create_model()
        # self.nmpcs_options.build_ocp_solver = False
        nmpc = nonlinear_MPC.NMPCAcados(setup_model, self.nmpcs_options)
        nmpc.set_reference(goal_pos, ees_pose)
        nmpc(z)   

    def test_closed_loop_performance(self, n_traj: int = 0):
        # Create controller
        setup_model = self._create_model()
        # self.nmpcs_options.build_ocp_solver = False
        nmpc = nonlinear_MPC.NMPCAcados(setup_model, self.nmpcs_options)

        # Create environment
        env = DLOEnv()

        # Run closed loop
        U = []
        z, fps_pos_des = env.reset(n_traj)
        nmpc.set_reference(fps_pos_des, z[3*nmpc.model.n_fps:])
        for k in range(25):
            u = nmpc(z)
            U.append(u)
            z = env.step(u)

        
        fps_pos = np.concatenate(env.fps_pos_history, axis=0)
        fps_pos_des = np.tile(fps_pos_des, (fps_pos.shape[0], 1))

        visualization.visualize_mass_spring_system(
            fps_pos, fps_pos_des, 0.1, 5
        )


class TestNMPCCasadi:
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

    def test_solve(self):
        n_traj = 0
        data = load_simulation_trajectory(n_traj)
        fps_pos = data[1, 1:31]
        ees_pose = data[1, 31:45]
        goal_pos = data[11, 87:117]

        z = np.concatenate((fps_pos, ees_pose))

        # Create controller
        setup_model = self._create_model()
        nmpc = nonlinear_MPC.NMPCCasadi(setup_model, self.nmpcs_options)

        # Solve
        u = nmpc(z, goal_pos)

    def test_closed_loop_performance(self, n_traj: int = 0):
        # Create controller
        setup_model = self._create_model()
        nmpc = nonlinear_MPC.NMPCCasadi(setup_model, self.nmpcs_options)

        # Create environment
        env = DLOEnv(n_traj)

        # Run closed loop
        U = []
        z, fps_pos_des = env.reset()
        for k in range(25):
            u = nmpc(z, fps_pos_des)
            U.append(u)
            z = env.step(u)

        
        fps_pos = np.concatenate(env.fps_pos_history, axis=0)
        fps_pos_des = np.tile(fps_pos_des, (fps_pos.shape[0], 1))

        visualization.visualize_mass_spring_system(
            fps_pos, fps_pos_des, 0.1, 5
        )


if __name__ == '__main__':
    # t = TestNMPCCasadi()
    # t.test_solve()
    # t.test_closed_loop_performance()

    t = TestNMPCAcados()
    # t.test_solve()
    t.test_closed_loop_performance(1)