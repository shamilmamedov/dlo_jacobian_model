from dataclasses import dataclass
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from scipy.linalg import block_diag
from tempfile import mkdtemp
import casadi as cs

import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model

@dataclass
class NMPCOptions:
    dt: float # sampling time aka step size
    N: int # prediction horizon
    u_max: np.ndarray
    u_min: np.ndarray
    build_ocp_solver: bool = True

    def __post_init__(self):
        self.tf = self.dt * self.N


class NMPCAcados:
    def __init__(
            self,
            model: casadi_dlo_model.DualArmDLOModel,
            options: NMPCOptions,
    ):
        self.model = model
        self.options = options

        self.iter_counter = 0
        acados_model = self._construct_acados_model()
        acados_ocp = self._construct_acados_ocp(acados_model)
        self.acados_ocp_solver = self._setup_acados_ocp_solver(acados_ocp)

    def _construct_acados_model(self) -> AcadosModel:
        xdot = cs.MX.sym('xdot', self.model.nx, 1)
        x, u, rhs = self.model._get_setup_dynamics_expr()

        acados_model = AcadosModel()
        acados_model.x = x
        acados_model.u = u
        acados_model.xdot = xdot
        acados_model.f_expl_expr = rhs
        acados_model.f_impl_expr = xdot - rhs
        acados_model.name = 'DualArmDLOModel'
        # acados_model.con_h_expr = ?
        # acados_model.con_h_expr_e = ?
        return acados_model
    
    def _construct_acados_ocp(self, acados_model: AcadosModel) -> AcadosOcp:
        """
        Acados docs on AcadosOcp: https://tinyurl.com/ywd6t3jp
        """
        ocp = AcadosOcp()
        ocp.model = acados_model
        ocp.dims.N = self.options.N

        # Defines cost function
        Q, R = self._cost_weights()
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = block_diag(Q, R)
        ocp.cost.W_e = np.copy(Q)

        n_vars = self.model.nx + self.model.nu
        ocp.cost.Vx = np.zeros((n_vars, self.model.nx))
        ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)

        ocp.cost.Vu = np.zeros((n_vars, self.model.nu))
        ocp.cost.Vu[self.model.nx:, :] = np.eye(self.model.nu)

        ocp.cost.Vx_e = np.eye(self.model.nx)

        # These are just placeholders, y_ref can be defined later
        ocp.cost.yref = np.zeros((n_vars,))
        ocp.cost.yref_e = np.zeros((self.model.nx,))

        # Defines constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.x0 = np.zeros((self.model.nx,))
        ocp.constraints.lbu = self.options.u_min
        ocp.constraints.ubu = self.options.u_max
        ocp.constraints.idxbu = np.arange(self.model.nu)
        return ocp

    def _cost_weights(self):
        nx = self.model.nx
        nu = self.model.nu
        q = np.ones((nx,))
        q[3*self.model.n_fps:] = 1e-2 # penalize end-effector pose deviation from the initial pose
        r = np.ones((nu,))

        Q = np.diag(q)
        R = np.diag(r)
        return Q, R

    def _setup_acados_ocp_solver(self, acados_ocp: AcadosOcp) -> AcadosOcpSolver:
        acados_ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        acados_ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        acados_ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        acados_ocp.solver_options.integrator_type = 'ERK'
        acados_ocp.solver_options.sim_method_num_stages = 1
        acados_ocp.solver_options.sim_method_num_steps = 1
        acados_ocp.solver_options.tf = self.options.tf
        
        acados_ocp.code_export_directory = mkdtemp()
        acados_ocp.code_export_directory = 'c_generated_code_dlo'
        acados_ocp_solver = AcadosOcpSolver(
            acados_ocp, 
            json_file='acados_ocp_dlo.json',
            build=self.options.build_ocp_solver,
            generate=self.options.build_ocp_solver,
        )
        # AcadosOcpSolver.generate(acados_ocp, json_file='acados_ocp_dlo.json')
        # AcadosOcpSolver.build(acados_ocp.code_export_directory, with_cython=True)
        # self.acados_ocp_solver = AcadosOcpSolver.create_cython_solver('acados_ocp_dlo.json')]
        # , build=False, generate=False
        return acados_ocp_solver

    def set_reference(self, desired_p_fps, current_poses):
        x_ref = np.concatenate((desired_p_fps, current_poses))
        u_ref = np.zeros((self.model.nu,))
        y_ref = np.concatenate((x_ref, u_ref))
        
        for stage in range(self.options.N):
            self.acados_ocp_solver.cost_set(stage, 'yref', y_ref)

        stage = self.options.N
        self.acados_ocp_solver.cost_set(stage, 'yref', x_ref)

    def _set_initial_guess(self, x):
        u0 = np.zeros((self.model.nu,))
        for stage in range(self.options.N):
            self.acados_ocp_solver.set(stage, 'x', x)
            self.acados_ocp_solver.set(stage, 'u', u0)
        
        self.acados_ocp_solver.set(self.options.N, 'x', x)

    def __call__(self, x):
        if self.iter_counter == 0:
            self._set_initial_guess(x)

        self.acados_ocp_solver.set(0, 'lbx', x)
        self.acados_ocp_solver.set(0, 'ubx', x)
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {} in time step {}".format(status, self.iter_counter))
        print("acados solved in {}".format(self.acados_ocp_solver.get_stats('time_tot')))

        self.iter_counter += 1
        if status == 0:
            return self.acados_ocp_solver.get(0, 'u')
        else: 
            self.acados_ocp_solver.reset()
            return np.zeros((self.model.nu,))
        


class NMPCCasadi:
    def __init__(
            self,
            model: casadi_dlo_model.DualArmDLOModel,
            options: NMPCOptions,
    ) -> None:
        self.model = model
        self.options = options

        self.n_fps = self.model.n_fps
        self.horizon = self.options.N
        self.nx = self.model.nx
        self.nx_fps = 3*self.n_fps
        self.nu = self.model.nu

        nlp = self._formulate_nlp()
        self.solver = self._setup_nlp_solver(nlp)

    def _formulate_nlp(self):
        x0, xd, X, U = self._symbolic_variables_for_states_and_controls()

        # Dynamics constraints
        g = []
        for i in range(self.horizon):
            # Euler integration
            rhs = X[:,i] + self.options.dt * self.model(X[:,i], U[:,i])
            g.append(X[:, i+1] - rhs)

        # Initial state constraint
        g.append(X[:,0] - x0)
        g = cs.vertcat(*g)

        # Objective function
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
        self.nw = w.shape[0]
        self.w0 = None
        self.lbw = -cs.inf*cs.DM.ones(self.nw)
        self.ubw = cs.inf*cs.DM.ones(self.nw)
        self.lbw[:self.nu*self.horizon] = cs.repmat(self.options.u_min, self.horizon, 1)
        self.ubw[:self.nu*self.horizon] = cs.repmat(self.options.u_max, self.horizon, 1)

        # Parameters vector
        p = cs.vertcat(x0, xd)

        # Define the NLP
        nlp = {'x': w, 'p': p, 'f': obj, 'g': g}
        return nlp
    
    def _symbolic_variables_for_states_and_controls(self):
        x0 = cs.MX.sym('x0', self.nx)
        # Desired positions of the feature points
        xd = cs.MX.sym('xd', self.nx_fps)
        # State variables
        X = cs.MX.sym('X', (self.nx, self.horizon + 1)) 
        # Control inputs
        U = cs.MX.sym('U', (self.nu, self.horizon))
        return x0, xd, X, U

    def _setup_nlp_solver(self, nlp):
        opts = {
            'ipopt': {
                'max_iter': 100,
                'print_level': 1,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
            },
            'print_time': 1,
        }
        solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
        return solver
    
    def _initial_guess(self, z):
        U0 = cs.DM.zeros((self.nu*self.horizon))
        X0 = cs.repmat(z, self.horizon+1, 1)
        w0 = cs.vertcat(U0, X0)
        return w0

    def __call__(self, z, xd):
        if self.w0 is None:
            self.w0 = self._initial_guess(z)

        p = cs.vertcat(z, xd)
        sol = self.solver(x0=self.w0, p=p, ubg=0., lbg=0., lbx=self.lbw, ubx=self.ubw)
        w_opt = sol['x']
        self.w0 = w_opt
        u = w_opt[:self.nu].full().flatten()
        return np.array(u)