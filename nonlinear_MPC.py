from dataclasses import dataclass
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from scipy.linalg import block_diag
from tempfile import mkdtemp
import casadi as cs

import casadi_dlo_model

@dataclass
class NMPCOptions:
    dt: float # sampling time aka step size
    N: int # prediction horizon
    u_max: np.ndarray
    u_min: np.ndarray
    build_ocp_solver: bool = True

    def __post_init__(self):
        self.tf = self.dt * self.N


class NMPC:
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
        acados_ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
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
        
        breakpoint()
        for stage in range(self.options.N):
            self.acados_ocp_solver.cost_set(stage, 'yref', y_ref)

        stage = self.options.N
        self.acados_ocp_solver.cost_set(stage, 'yref', x_ref)

    def __call__(self, x):
        self.acados_ocp_solver.set(0, 'lbx', x)
        self.acados_ocp_solver.set(0, 'ubx', x)
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {} in time step {}".format(status, self.iter_counter))

        breakpoint()
        self.iter_counter += 1
        if status == 0:
            return self.acados_ocp_solver.get(0, 'u')
        else: 
            self.acados_ocp_solver.reset()
            return np.zeros((self.model.nu,))
        
