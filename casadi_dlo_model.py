import casadi as cs
import numpy as np
from typing import Union
from dataclasses import dataclass

from RBF import JacobianPredictor 
import casadi_rbf
import casadi_nn


class JacobianNetwork:
    def __init__(
            self,
            n_feature_points: int,
            n_hidden_units: int,
            rbf_centers: cs.DM,
            rbf_sigmas: cs.DM,
            lin_A: cs.DM,
            lin_b: cs.DM = None,
            jit: bool = True
    ):
        """
        NOTE the input is relative positions and orientation; thus,
             3 is the raltive positions between the end-effectors
             4+4 are quaternions of two arms

             12 is the number of inputs -- velocities of the end-effectors
        """
        self.n_fps = n_feature_points
        self.n_hu = n_hidden_units
        lw = [3*self.n_fps + 3+4+4, self.n_hu, (self.n_fps * 3) * 12]
        
        assert rbf_centers.shape == (lw[1], lw[0])
        assert lin_A.shape == (lw[2], lw[1])

        basis_fcn = casadi_rbf.gaussian
        self.rbf = casadi_rbf.RBF(rbf_centers, rbf_sigmas, basis_fcn)
        self.linear = casadi_nn.Linear(lin_A, lin_b)

        self._length_invariant_jacobian_fcn = self._get_length_invariant_jacobian_fcn()

    def _get_length_invariant_jacobian_expr(self):
        x_abs = cs.MX.sym('x', (3*self.n_fps + 14, 1))
        x_rel = self._compute_relative_positions(x_abs)
        theta = self.rbf(x_rel)
        out =  self.linear(theta)
        out = self._reshape_vec2mat(out)
        return x_abs, out

    def _get_length_invariant_jacobian_fcn(self):
        x_abs, out = self._get_length_invariant_jacobian_expr()
        J = cs.Function('J', [x_abs], [out])
        return J
    
    def _compute_relative_positions(self, abs_pos: Union[cs.DM, cs.SX]):
        # Parse absolute positions and orientation
        n_fps = self.n_fps
        left_end_pos = abs_pos[3*n_fps : 3*n_fps + 3]
        left_end_quat = abs_pos[3*n_fps + 3 : 3*n_fps +7]
        right_end_pos = abs_pos[3*n_fps + 7 : 3*n_fps +10]
        right_end_quat = abs_pos[3*n_fps +10 : 3*n_fps +14]
        
        # Compute ralative normalized positions of feature points
        fps_pos = abs_pos[0:3*n_fps].reshape((3,n_fps))
        fps_pos_rel = type(abs_pos).zeros(fps_pos.shape)
        fps_pos_rel[:,1:] = cs.diff(fps_pos, 1, 1)
        for fp in range(1,n_fps):
            norm_ = cs.norm_2(fps_pos_rel[:,fp]) 
            fps_pos_rel[:,fp] /= (norm_ + 1e-6)
        fps_pos_rel = fps_pos_rel.reshape((-1,1))

        # Compute relative positions of the robot end effectors
        right_end_pos_rel = (right_end_pos - left_end_pos) 
        right_end_pos_rel /= (cs.norm_2(right_end_pos_rel) + 1e-6)
        c_rel = cs.vertcat(fps_pos_rel, right_end_pos_rel, left_end_quat, right_end_quat)
        return c_rel
    
    def _reshape_vec2mat(self, vec):
        vec = vec.reshape((-1, self.n_fps))
        vec_cols = cs.horzsplit_n(vec, self.n_fps)
        mat = cs.vertcat(
            *[col.reshape((-1,12)) for col in vec_cols]
        )
        return mat

    def compute_length_invariant_jacobian(self, x):
        return self._length_invariant_jacobian_fcn(x)
    
    def __call__(self, x, dlo_length: float) -> cs.DM:
        """
        :param x: the absolute positions and orientations of the end-effectors
        :param dlo_length: the length of the DLO

        :return: the jacobian of the feature points wrt the end-effector velocities
        """
        J = self.compute_length_invariant_jacobian(x)
        J[:, [3, 4, 5, 9, 10, 11]] *= dlo_length
        return J
       
    def get_feature_points_velocity_fcn(self):
        x_abs, u, p, fps_vel = self.get_feature_points_velocities_expr()
        dx = cs.Function('dx', [x_abs, u, p], [fps_vel])
        return dx

    def get_feature_points_velocities_expr(self):
        """
        "states" are the absolute positions and orientations of the end-effectors
        "inputs" are the velocities of the end-effectors
        "parameters" are the parameters of the RBF and linear layers

        """
        # Length invariant jacobian
        x_abs, J = self._get_length_invariant_jacobian_expr()
        
        # The infuluence of length on the jacobian
        l = cs.MX.sym('l')
        J[:, [3, 4, 5, 9, 10, 11]] *= l

        # Compute fps velocities
        u = cs.MX.sym('u', (12,1))
        fps_vel = J @ u

        # Parametes
        p = l
        return x_abs, u, p, fps_vel


class DualArmDLOModel:
    """ Implements complete model of the DLO attached to the two end-effectors
    the model states are: feature point positions, end-effector poses
    the model inputs are: end-effector velocities
    """
    def __init__(
            self,
            dlo_model: JacobianNetwork,
            dlo_length: float,
    ) -> None:
        self.dlo_model = dlo_model
        self.dlo_length = dlo_length
        self.n_fps = self.dlo_model.n_fps
        self.nx = 3*self.dlo_model.n_fps + 14
        self.nu = 12

        # JIT ME !!!!!!!!!!!
        self._setup_dynamics_fcn = self._get_setup_dynamics_fcn()

    def _get_dual_arm_dynamics_expr(self):
        poses = cs.MX.sym('poses', 14, 1)
        vels = cs.MX.sym('vels', 12, 1)

        # Compute the end-effector pose dynamics
        arm1 = end_effector_pose_dynamics(poses[0:7], vels[0:6])
        arm2 = end_effector_pose_dynamics(poses[7:14], vels[6:12])
        poses_dot = cs.vertcat(arm1, arm2)
        return poses, vels, poses_dot

    def _get_setup_dynamics_expr(self):
        poses, vels, poses_dot = self._get_dual_arm_dynamics_expr()
        dlo_pos = cs.MX.sym('x', (3*self.dlo_model.n_fps, 1)) 

        # state
        x_abs = cs.vertcat(dlo_pos, poses)

        # dlo velocity
        dlo_pos_dot = self.dlo_model(x_abs, self.dlo_length) @ vels

        # setup dynamics
        x_dot = cs.vertcat(dlo_pos_dot, poses_dot)

        return x_abs, vels, x_dot
    
    def _get_setup_dynamics_fcn(self):
        z, u, z_dot = self._get_setup_dynamics_expr()
        dz = cs.Function('dz', [z, u], [z_dot])
        return dz
    
    def __call__(self, x, u):
        return self._setup_dynamics_fcn(x, u)
    
    def _get_linearized_setup_dynamics_expr(self):
        z, u, x_dot = self._get_setup_dynamics_expr()
        A = cs.jacobian(x_dot, z)
        B = cs.jacobian(x_dot, u)
        return z, u, A, B
    
    def _get_linearized_setup_dynamics_fcns(self, jit: bool = False):
        opts = {}
        if jit:
            opts = {'jit': True, 'compiler': 'shell', 'verbose': True,
                    'jit_options': {'flags': '-O3'}} 

        z, u, A, B = self._get_linearized_setup_dynamics_expr()
        A_fcn = cs.Function('A', [z, u], [A], opts)
        B_fcn = cs.Function('B', [z, u], [B], opts)
        return A_fcn, B_fcn


def quaternion_derivative_jacobian_wrt_q(w):
    out = type(w).zeros(4,4)
    out[:3,:3] = -cs.skew(w)
    out[:3,3] = w
    out[3,:3] = -w.T
    return out


def quaternion_derivative_jacobian_wrt_w(q):
    x, y, z, w = cs.vertsplit_n(q,4)
    out = type(q).zeros(4,3)
    out[0,:] = [w, -z, y]
    out[1,:] = [z, w, -z]
    out[2,:] = [-y, x, w]
    out[3,:] = [-x, -y, -z]
    return out


def end_effector_pose_dynamics(pose: cs.DM, vel: cs.DM) -> cs.DM:
    """
    Compute the end-effector pose dynamics
    """
    # Parse the pose
    pos = pose[0:3]
    quat = pose[3:7]

    # Parse the velocity
    vel_lin = vel[0:3]
    vel_ang = vel[3:6]

    # Compute the pose derivative
    quat_dot = 0.5 * quaternion_derivative_jacobian_wrt_q(vel_ang) @ quat
    # quat_dot /= cs.norm_2(quat_dot)
    pos_dot = vel_lin

    # Return the pose derivative
    return cs.vertcat(pos_dot, quat_dot)
    

def load_model_parameters():
    # Instantiate Yu's jacobian predictor and load weights
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = cs.DM(jp.model_J.fc1.centres.cpu().detach().numpy())
    rbf_sigmas = cs.DM(jp.model_J.fc1.sigmas.cpu().detach().numpy())
    lin_A = cs.DM(jp.model_J.fc2.weight.cpu().detach().numpy())

    # Instantiate casadi implementation
    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden

    return {'n_feature_points': n_feature_points,
            'n_hidden_units': n_hidden_units,
            'rbf_centers': rbf_centers,
            'rbf_sigmas': rbf_sigmas,
            'lin_A': lin_A}