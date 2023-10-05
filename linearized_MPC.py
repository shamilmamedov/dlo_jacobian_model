from typing import Union
from dataclasses import dataclass
import casadi as cs
import numpy as np


import casadi_dlo_model


@dataclass
class LinearizedMPCOptions:
    dt: float # sampling time aka step size
    N: int # prediction horizon
    u_max: Union[cs.DM, np.ndarray]
    u_min: Union[cs.DM, np.ndarray]


class LinearizedMPC:
    def __init__(
            self,
            model: casadi_dlo_model.DualArmDLOModel,
            options: LinearizedMPCOptions,
    ) -> None:
        self.model = model
        self.options = options

        self.n_fps = self.model.n_fps
        self.horizon = self.options.N
        self.nx = self.model.nx
        self.nu = self.model.nu

        self._A_fcn, self._B_fcn = self.model._get_linearized_setup_dynamics_fcns()

        qp = self._formulate_mpc_as_qp()
        self.solver = self._setup_qp_solver(qp, 'qpoases') #qpOASES

    def _formulate_mpc_as_qp(self):
        delta_x_des, delta_U = self._symbolic_variables_for_states_and_controls()
        A, B = self._symbolic_variables_for_dynamics()

        # Dynamics propagation
        delta_X = cs.MX.zeros(self.nx, self.horizon + 1)
        for i in range(self.horizon):
            delta_X[:,i+1] = A @ delta_X[:,i] + B @ delta_U[:,i]

        # Objective function including the terminal cost
        obj = 0
        for i in range(self.horizon):
            output_error = delta_X[:3*self.n_fps,i+1] - delta_x_des
            error_arm1 = delta_X[3*self.n_fps:3*self.n_fps+3,i+1] - delta_x_des[:3]
            error_arm2 = delta_X[3*self.n_fps+7:3*self.n_fps+10,i+1] - delta_x_des[-3:]
            obj += (
                output_error.T @ output_error + 
                error_arm1.T @ error_arm1 + 
                error_arm2.T @ error_arm2 +
                1e+2*delta_U[:,i].T @ delta_U[:,i]
            )
        # Intial state constraint
        g = delta_X[:,0]

        # Decision variables
        w = cs.vec(delta_U)

        # Parameters vector
        p = cs.vertcat(
            delta_x_des,
            cs.vec(A),
            cs.vec(B)
        )

        # Define the QP
        qp = {'x': w, 'p': p, 'f': obj, 'g': g}
        return qp
    
    def _symbolic_variables_for_states_and_controls(self):
        # Desired positions of the feature points
        delta_x_des = cs.MX.sym('delta_x_des', 3*self.n_fps)

        # Control inputs
        delta_U = cs.MX.sym('delta_U', (self.nu, self.horizon))
        
        return delta_x_des, delta_U

    def _symbolic_variables_for_dynamics(self):
        A = cs.MX.sym('A', (self.nx, self.nx))
        B = cs.MX.sym('B', (self.nx, self.nu))
        return A, B
    
    def _setup_qp_solver(self, qp, solver: str):
        return cs.qpsol('S', solver, qp)
    
    def _get_linearized_dynamics(self, z, u):
        A = self._A_fcn(z, u)
        B = self._B_fcn(z, u)

        # Discretize the linearized dynamics with the explicit Euler method
        A_discr = cs.DM.eye(self.nx) + self.options.dt * A
        B_discr = self.options.dt * B
        return A_discr, B_discr

    def _solve(self, z, u, x_des):
        A, B = self._get_linearized_dynamics(z, u)

        delta_x_des = x_des - z[:3*self.n_fps]
        # delta_x_des = cs.DM.zeros(3*self.n_fps)
        p = cs.vertcat(
            delta_x_des,
            cs.vec(A),
            cs.vec(B)
        )
        sol = self.solver(x0=0, p=p, lbg=0, ubg=0)#, lbx=-0.01, ubx=0.01)
        delta_u = sol['x'].full().reshape(-1, self.nu)
        return delta_u

    def __call__(self, z, u, x_des):
        delta_u = self._solve(z, u, x_des)
        return u + delta_u[0]