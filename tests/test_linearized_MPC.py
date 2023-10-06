import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model
from dlo_jacobian_model.linearized_MPC import LinearizedMPC, LinearizedMPCOptions
import dlo_jacobian_model.visualization as visualization
from dlo_jacobian_model.env import DLOEnv


class TestLinearizedMPC:
    lmpc_opts = LinearizedMPCOptions(
        dt=0.1,
        N=10,
        u_max=cs.DM.ones(12),
        u_min=-cs.DM.ones(12),
    )

    def _create_model(self):
        dlo_model_parms = casadi_dlo_model.load_model_parameters()
        dlo_model = casadi_dlo_model.JacobianNetwork(**dlo_model_parms)
        dlo_length = 0.5
        setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)
        return setup_model

    def test_closed_loop_performance(self, n_traj: int = 0):
        model = self._create_model()
        lmpc = LinearizedMPC(model, self.lmpc_opts)

        env = DLOEnv()
        U = []
        z, fps_pos_des = env.reset(n_traj)
        for k in range(25):
            u = lmpc(z, fps_pos_des)
            U.append(u)
            z = env.step(u)

        U = np.vstack(U)
        u_lin_l = U[:,:3]
        u_ang_l = U[:,3:6]
        u_lin_r = U[:,6:9]
        u_ang_r = U[:,9:12]
        t = np.arange(0, U.shape[0])*self.lmpc_opts.dt

        _, axs = plt.subplots(3,1, sharex=True)
        axs[0].plot(t, u_lin_l[:,0])
        axs[0].plot(t, u_lin_r[:,0])
        axs[1].plot(t, u_lin_l[:,1])
        axs[1].plot(t, u_lin_r[:,1])
        axs[2].plot(t, u_lin_l[:,2])
        axs[2].plot(t, u_lin_r[:,2])
        plt.tight_layout()

        _, axs = plt.subplots(3,1, sharex=True)
        axs[0].plot(t, u_ang_l[:,0])
        axs[0].plot(t, u_ang_r[:,0])
        axs[1].plot(t, u_ang_l[:,1])
        axs[1].plot(t, u_ang_r[:,1])
        axs[2].plot(t, u_ang_l[:,2])
        axs[2].plot(t, u_ang_r[:,2])
        plt.tight_layout()
        # plt.show()


        fps_pos = np.concatenate(env.fps_pos_history, axis=0)
        fps_pos_des = np.tile(fps_pos_des, (fps_pos.shape[0], 1))

        visualization.visualize_mass_spring_system(
            fps_pos, fps_pos_des, 0.1, 5
        )


if __name__ == '__main__':
    t = TestLinearizedMPC()
    t.test_closed_loop_performance()
    # t.test_solve()


