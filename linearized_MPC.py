
import casadi as cs


class LinearizedMPC:
    def __init__(self, n_feature_points: int, horizon: int) -> None:
        self.n_fps = n_feature_points
        self.horizon = horizon

        self.nx = 3 * self.n_fps + 2*7
        self.nu = 12

        qp = self._formulate_mpc_as_qp()
        self.solver = self._setup_qp_solver(qp, 'qpoases')

    def _formulate_mpc_as_qp(self):
        x0, x_des, u = self._symbolic_variables_for_states_and_controls()
        A, B = self._symbolic_variables_for_dynamics()

        # Dynamics propagation
        x = cs.SX.zeros(self.nx, self.horizon + 1)
        x[:,0] = x0
        for i in range(self.horizon):
            x[:,i+1] = A @ x[:,i] + B @ u[:,i]

        # Objective function
        obj = 0
        for i in range(self.horizon):
            obj += cs.sumsqr(x_des - x[:3*self.n_fps,i])

        # Decision variables
        w = cs.vec(u)

        # Parameters vector
        p = cs.vertcat(
            x0,
            x_des,
            cs.vec(A),
            cs.vec(B)
        )

        # Define the QP
        qp = {'x': w, 'p': p, 'f': obj}
        return qp
    
    def _symbolic_variables_for_states_and_controls(self):
        # Initial state
        x0 = cs.SX.sym('x0', self.nx)

        # Desired positions of the feature points
        x_des = cs.SX.sym('x_des', 3*self.n_fps)

        # Control inputs
        u = cs.SX.sym('u', (self.nu, self.horizon))
        
        return x0, x_des, u

    def _symbolic_variables_for_dynamics(self):
        A = cs.SX.sym('A', (self.nx, self.nx))
        B = cs.SX.sym('B', (self.nx, self.nu))
        return A, B
    
    def _setup_qp_solver(qp, solver: str):
        return cs.qpsol('S', solver, qp)
    
    def optimize_control_inputs(self, x0, x_des, A, B):
        p = cs.vertcat(
            x0,
            x_des,
            cs.vec(A),
            cs.vec(B)
        )
        sol = self.solver(x0=0, p=p)
        return sol['x'].full().reshape(-1, self.nu).T