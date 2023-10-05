from typing import Union
from dataclasses import dataclass
import casadi as cs
import numpy as np

import casadi_dlo_model


@dataclass
class RuansMPCOptions:
    dt: float
    N: int
    u_max: Union[cs.DM, np.ndarray]
    u_min: Union[cs.DM, np.ndarray]


class RuansMPC:
    def __init__(
            self,
            dlo_model: casadi_dlo_model.JacobianNetwork,
            options: RuansMPCOptions,
    ):
        self.model = dlo_model
        self.options = options

        self.n_fps = self.model.n_fps
        self.horizon = self.options.N
        self.nx = 3*self.n_fps
        self.nu = 12

        self.solver = self._formulate_mpc_as_qp()

    def _formulate_mpc_as_qp(self):
        x0, xd, X, U = self._symbolic_vars_for_states_and_controls()
        B = self._symbolic_vars_for_dynamics()

        # Multiple shooring dynamics constraints
        g = []
        for i in range(self.horizon):
            g.append(X[:,i+1] - X[:,i] - self.options.dt*B @ U[:,i])        

        # Initial state constraint
        g.append(X[:,0] - x0)       
        g = cs.vertcat(*g)

        # Objective function
        obj = 0.
        for i in range(self.horizon):
            state_err = X[:,i+1] - xd
            obj += U[:, i].T @ U[:, i]  + state_err.T @ state_err 

        # Decision variables
        w = cs.vertcat(
            cs.vec(U),
            cs.vec(X)
        )

        # Bounds on decision variables        
        nw = w.shape[0]
        self.lbx = np.array([-np.inf]*nw)
        self.ubx = np.array([np.inf]*nw)
        self.lbx[:self.nu*self.horizon] = -0.1
        self.ubx[:self.nu*self.horizon] = 0.1
       
        # Parameters vector
        p = cs.vertcat(x0, xd, cs.vec(B))

        # Define the QP
        qp = {'x': w, 'f': obj, 'g': g, 'p': p}

        print("Creating solver...")
        # Create a solver instance
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-2}
        # solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-2}
        # opts = {'print_iter': True, 'print_time': 1}
        solver = cs.qpsol('solver', 'qrqp', qp)#, opts)
        print("Solver successfully created! :)")
        return solver
    
    def _symbolic_vars_for_states_and_controls(self):
        # current position of fps
        x0 = cs.SX.sym('x0', self.nx)
        # desired position of fps
        xd = cs.SX.sym('xd', self.nx)
        # state decision variables
        X = cs.SX.sym('X', (self.nx, self.horizon+1))
        # control decision variables
        U = cs.SX.sym('U', (self.nu, self.horizon))
        return x0, xd, X, U

    def _symbolic_vars_for_dynamics(self):
        # input-to-state map
        B = cs.SX.sym('B', (self.nx, self.nu))
        return B

    def _cost_function_weights(self):
        r_lin = 1.
        r_ang = 1.
        R = np.diag(
            [r_lin]*3 + [r_ang]*3 +
            [r_lin]*3 + [r_ang]*3
        )
        return R
    
    def __call__(self, z, xd, dlo_length: float = 0.5):
        B = self.model(z, dlo_length)
        x0 = z[:self.nx]
        p = cs.vertcat(x0, xd, cs.vec(B))
        sol = self.solver(
            x0=0, ubg=0, lbg=0,
            ubx=self.ubx, lbx=self.lbx, p=p
        )
        opt_decision_vars = sol['x']
        u = opt_decision_vars[:12].full().flatten()

        return u