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
        self.solver = self._setup_qp_solver(qp, 'qrqp') #qpOASES

    def _formulate_mpc_as_qp(self):
        x_des, u = self._symbolic_variables_for_states_and_controls()
        A, B = self._symbolic_variables_for_dynamics()

        # Dynamics propagation
        x = cs.MX.zeros(self.nx, self.horizon + 1)
        for i in range(self.horizon):
            x[:,i+1] = A @ x[:,i] + B @ u[:,i]

        # Objective function
        obj = 0
        for i in range(self.horizon):
            obj += (
                cs.sumsqr(x_des - x[:3*self.n_fps,i]) +
                cs.sumsqr(x[3*self.n_fps:,i]) +
                cs.sumsqr(u[:,i])
            )

        obj += cs.sumsqr(x_des - x[:3*self.n_fps, self.horizon]) # terminal cost

        # Decision variables
        w = cs.vec(u)

        # Parameters vector
        p = cs.vertcat(
            x_des,
            cs.vec(A),
            cs.vec(B)
        )

        # Define the QP
        qp = {'x': w, 'p': p, 'f': obj}
        return qp
    
    def _symbolic_variables_for_states_and_controls(self):
        # Desired positions of the feature points
        x_des = cs.MX.sym('x_des', 3*self.n_fps)

        # Control inputs
        u = cs.MX.sym('u', (self.nu, self.horizon))
        
        return x_des, u

    def _symbolic_variables_for_dynamics(self):
        A = cs.MX.sym('A', (self.nx, self.nx))
        B = cs.MX.sym('B', (self.nx, self.nu))
        return A, B
    
    def _setup_qp_solver(self, qp, solver: str):
        return cs.qpsol('S', solver, qp)
    
    def _get_linearized_dynamics(self, x, u):
        A = self._A_fcn(x, u)
        B = self._B_fcn(x, u)

        # Discretize the linearized dynamics with the explicit Euler method
        A_discr = cs.DM.eye(self.nx) + self.options.dt * A
        B_discr = self.options.dt * B
        return A_discr, B_discr

    def _solve(self, x, u, x_des):
        A, B = self._get_linearized_dynamics(x, u)
        x_des = x_des - x[:3*self.n_fps]
        p = cs.vertcat(
            x_des,
            cs.vec(A),
            cs.vec(B)
        )
        sol = self.solver(x0=0, p=p)
        delta_u = sol['x'].full().reshape(-1, self.nu)
        return delta_u

    def __call__(self, x, u, x_des):
        delta_u = self._solve(x, u, x_des)
        return u + delta_u[0]