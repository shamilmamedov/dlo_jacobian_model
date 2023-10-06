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
    solver: str = 'qrqp'


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
        self.nx_fps = 3*self.n_fps
        self.nu = self.model.nu

        self._A_fcn, self._B_fcn = self.model._get_linearized_setup_dynamics_fcns()

        qp = self._formulate_mpc_as_qp()
        self.solver = self._setup_qp_solver(qp) #qpOASES
        self.u = cs.DM.zeros(self.nu)

    def _formulate_mpc_as_qp(self):
        x0, xd, X, U = self._symbolic_variables_for_states_and_controls()
        A, B, c = self._symbolic_variables_for_dynamics()

        # Dynamics propagation
        g = []
        for i in range(self.horizon):
            rhs = A @ X[:,i] + B @ U[:,i] + c
            g.append(X[:,i+1] - rhs)

        # Initial state constraint
        g.append(X[:,0] - x0)
        g = cs.vertcat(*g)

        # Objective function including the terminal cost
        obj = 0
        for i in range(self.horizon):
            fps_pos_err = X[:self.nx_fps, i+1] - xd
            l_arm_pos_err = X[self.nx_fps:self.nx_fps+3, i+1] - xd[:3]
            r_arm_pos_err = X[self.nx_fps+7:self.nx_fps+10, i+1] - xd[-3:]
            obj += (
                fps_pos_err.T @ fps_pos_err + 
                l_arm_pos_err.T @ l_arm_pos_err +
                r_arm_pos_err.T @ r_arm_pos_err +
                U[:,i].T @ U[:,i]
            )
        
        # Decision variables
        w = cs.vertcat(cs.vec(U), cs.vec(X))

        # Bounds on decision variables
        nw = w.shape[0]
        self.lbw = -cs.inf*cs.DM.ones(nw)
        self.ubw = cs.inf*cs.DM.ones(nw)
        self.lbw[:self.nu*self.horizon] = cs.repmat(self.options.u_min, self.horizon, 1)
        self.ubw[:self.nu*self.horizon] = cs.repmat(self.options.u_max, self.horizon, 1)

        # Parameters vector
        p = cs.vertcat(
            x0,
            xd,
            cs.vec(A),
            cs.vec(B),
            c
        )

        # Define the QP
        qp = {'x': w, 'p': p, 'f': obj, 'g': g}
        return qp
    
    def _symbolic_variables_for_states_and_controls(self):
        x0 = cs.MX.sym('x0', self.nx)
        # Desired positions of the feature points
        xd = cs.MX.sym('xd', self.nx_fps)
        # State variables
        X = cs.MX.sym('X', (self.nx, self.horizon + 1)) 
        # Control inputs
        U = cs.MX.sym('U', (self.nu, self.horizon))
        
        return x0, xd, X, U

    def _symbolic_variables_for_dynamics(self):
        A = cs.MX.sym('A', (self.nx, self.nx))
        B = cs.MX.sym('B', (self.nx, self.nu))
        c = cs.MX.sym('c', self.nx)
        return A, B, c
    
    def _setup_qp_solver(self, qp):
        opts = {'print_iter': True, 'print_time': 1}
        return cs.qpsol('S', self.options.solver, qp, opts)
    
    def _get_linearized_dynamics(self, z, u):
        A = self._A_fcn(z, u)
        B = self._B_fcn(z, u)
        c = self.model(z, u) - A @ z - B @ u

        # Discretize the linearized dynamics with the explicit Euler method
        A_discr = cs.DM.eye(self.nx) + self.options.dt * A
        B_discr = self.options.dt * B
        c_discr = self.options.dt * c
        return A_discr, B_discr, c_discr

    def __call__(self, z, xd):
        A, B, c = self._get_linearized_dynamics(z, self.u)

        p = cs.vertcat(
            z,
            xd,
            cs.vec(A),
            cs.vec(B),
            c
        )
        sol = self.solver(x0=0, p=p, ubg=0, lbg=0, ubx=self.ubw, lbx=self.lbw)
        w_opt = sol['x']
        self.u = w_opt[:self.nu].full().flatten()
        return np.array(self.u)


class LinearizedMPCWrapper:
    def __init__(self) -> None:
        opts = LinearizedMPCOptions(
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

        self.lmpc = LinearizedMPC(setup_model, opts)

    def generateControlInput(self, state):
        z, xd = self._parse_state(state)
        u = self.mpc(z, xd)
        return u

    def _parse_state(self, state):
        z = state[1:45]
        xd = state[87:117]
        return z, xd