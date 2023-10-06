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
    solver: str = 'qrqp'


class RuansMPC:
    def __init__(
            self,
            setup_model: casadi_dlo_model.DualArmDLOModel,
            options: RuansMPCOptions,
    ):
        self.model = setup_model
        self.options = options

        self.n_fps = self.model.n_fps
        self.horizon = self.options.N
        self.nx = 3*self.n_fps
        self.nu = 12

        self._A_fcn, self._B_fcn = self.model._get_linearized_setup_dynamics_fcns()
        self.solver = self._formulate_mpc_as_qp()
        self.u = cs.DM.zeros(self.nu)

    def _formulate_mpc_as_qp(self):
        x0, xd, X, U = self._symbolic_vars_for_states_and_controls()
        A, B = self._symbolic_vars_for_dynamics()

        # Multiple shooring dynamics constraints
        dt = self.options.dt
        I_nx = cs.DM.eye(self.nx)
        g = []
        for i in range(self.horizon):
            rhs = (I_nx + dt*A) @ X[:,i] + dt*B @ U[:,i] - dt*A @ x0
            g.append(X[:,i+1] - rhs)        

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
        self.lbx = -cs.inf*cs.DM.ones(nw)
        self.ubx = cs.inf*cs.DM.ones(nw)
        self.lbx[:self.nu*self.horizon] = cs.repmat(self.options.u_min, self.horizon, 1)
        self.ubx[:self.nu*self.horizon] = cs.repmat(self.options.u_max, self.horizon, 1)
        self.nw = nw
       
        # Parameters vector
        p = cs.vertcat(x0, xd, cs.vec(B), cs.vec(A))

        # Define the QP
        qp = {'x': w, 'f': obj, 'g': g, 'p': p}
        opts = {'print_iter': True, 'print_time': 1}
        solver = cs.qpsol('solver', self.options.solver, qp, opts)
        return solver
    
    def _symbolic_vars_for_states_and_controls(self):
        # current position of fps
        x0 = cs.SX.sym('x0', self.nx)
        # desired position of fps
        xd = cs.SX.sym('xd', self.nx)
        # state decision variables
        X = cs.SX.sym('X', (self.nx, self.horizon+1))# Intial state constraint

        # control decision variables
        U = cs.SX.sym('U', (self.nu, self.horizon))
        return x0, xd, X, U

    def _symbolic_vars_for_dynamics(self):
        # input-to-state map
        B = cs.SX.sym('B', (self.nx, self.nu))
        A = cs.SX.sym('A', (self.nx, self.nx))
        return A, B

    def _cost_function_weights(self):
        r_lin = 1.
        r_ang = 1.
        R = np.diag(
            [r_lin]*3 + [r_ang]*3 +
            [r_lin]*3 + [r_ang]*3
        )
        return R
    
    def _compute_linearized_dynamics(self, z, u):
        A = self._A_fcn(z, u)
        B = self._B_fcn(z, u)
        return A, B

    def __call__(self, z, xd):
        A, B = self._compute_linearized_dynamics(z, self.u)
        B = B[:self.nx, :self.nu]
        A = A[:self.nx, :self.nx]

        x0 = z[:self.nx]
        p = cs.vertcat(x0, xd, cs.vec(B), cs.vec(A))
        sol = self.solver(
            x0=0, ubg=0, lbg=0,
            ubx=self.ubx, lbx=self.lbx, p=p
        )
        opt_decision_vars = sol['x']
        self.u = opt_decision_vars[:12].full().flatten()

        return np.array(self.u)
    

class RuansMPCWrapper:
    def __init__(self) -> None:
        opts = RuansMPCOptions(
            dt=0.1,
            N=10,
            u_max=cs.DM.ones(12),
            u_min=-cs.DM.ones(12),
            solver='qrqp'
        )

        dlo_model_parms = casadi_dlo_model.load_model_parameters()
        dlo_model = casadi_dlo_model.JacobianNetwork(**dlo_model_parms)
        dlo_length = 0.5
        setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)

        self.mpc = RuansMPC(setup_model, opts)

    def generateControlInput(self, state):
        z, xd = self._parse_state(state)
        u = self.mpc(z, xd)
        return u

    def _parse_state(self, state):
        z = state[1:45]
        xd = state[87:117]
        return z, xd