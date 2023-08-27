import casadi as cs
import numpy as np
from typing import Union

import casadi_rbf
import casadi_nn


class JacobianNetwork:
    def __init__(
            self,
            n_feature_points: int,
            n_hidden_units: int
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
        
        basis_fcn = casadi_rbf.gaussian
        self.rbf = casadi_rbf.RBF(lw[0], lw[1], basis_fcn)
        self.linear = casadi_nn.Linear(lw[1], lw[2], bias=False)

    def compute_length_invariant_jacobian(self, x, rbf_centers, rbf_sigmas, lin_A, lin_b=None):
        x = self._compute_relative_positions(x)

        theta = self.rbf(x, rbf_centers, rbf_sigmas)
        out =  self.linear(theta, lin_A, lin_b)
        return self._reshape_vec2mat(out)

    def __call__(self, x, dlo_length, rbf_centers, rbf_sigmas, lin_A, lin_b=None):
        J = self.compute_length_invariant_jacobian(x, rbf_centers, rbf_sigmas, lin_A, lin_b)
        J[:, [3, 4, 5, 9, 10, 11]] *= dlo_length
        
        return J
       
    def _reshape_vec2mat(self, vec):
        vec = vec.reshape((-1, self.n_fps))
        vec_cols = cs.horzsplit_n(vec, self.n_fps)
        mat = cs.vertcat(
            *[col.reshape((-1,12)) for col in vec_cols]
        )
        return mat

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
            fps_pos_rel[:,fp] /= norm_
        fps_pos_rel = fps_pos_rel.reshape((-1,1))

        # Compute relative positions of the robot end effectors
        right_end_pos_rel = (right_end_pos - left_end_pos) 
        right_end_pos_rel /= cs.norm_2(right_end_pos_rel)
        c_rel = cs.vertcat(fps_pos_rel, right_end_pos_rel, left_end_quat, right_end_quat)
        return c_rel

    def get_feature_points_velocity_fcn(self):
        abs_poses, u, p, fps_vel = self.get_feature_points_velocities_expr()
        dx = cs.Function('dx', [abs_poses, u, p], [fps_vel])
        return dx

    def get_feature_points_velocities_expr(self):
        # Symbolic variables for inputs
        n_fps = self.n_fps
        abs_poses = cs.SX.sym('abs_poses', (3*n_fps + 14, 1))

        # Computing relative positions
        rel_poses = self._compute_relative_positions(abs_poses)

        # Computing length independent jacobian
        _, rbf_p, rbf_out = self.rbf._get_symbolic_rbf_expression(rel_poses)
        _, lin_p, lin_out = self.linear._get_symbolic_expression(rbf_out)

        J = self._reshape_vec2mat(lin_out)
        
        # The infuluence of length
        l = cs.SX.sym('l')
        J[:, [3, 4, 5, 9, 10, 11]] *= l

        # Compute fps velocities
        u = cs.SX.sym('u', (12,1))
        fps_vel = J @ u

        p = cs.vertcat(l, rbf_p, lin_p)
        return abs_poses, u, p, fps_vel


class QuasiStaticDLOModel:
    pass




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
    return np.concatenate((pos_dot, quat_dot))
    


def main():
    n_feature_points = 10
    n_hidden_units = 256
    m = JacobianNetwork(n_feature_points, n_hidden_units)

    z, u, p, dx = m.get_feature_points_velocities_expr()

    A_expr = cs.jacobian(dx, z)
    print(A_expr.shape)



if __name__ == '__main__':
    main()
