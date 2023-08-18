import casadi as cs
import numpy as np

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

    def __call__(self, x, rbf_centers, rbf_sigmas, lin_A, lin_b=None):
        x = self._compute_relative_positions(x)

        theta = self.rbf(x, rbf_centers, rbf_sigmas)
        out = self.linear(theta, lin_A, lin_b)

        out = out.reshape((-1, self.n_fps))
        out_cols = cs.horzsplit_n(out, self.n_fps)
        out = cs.vertcat(
            *[col.reshape((-1,12)) for col in out_cols]
        )
        return out

    def _compute_relative_positions(self, abs_pos: cs.DM):
        # Parse absolute positions and orientation
        n_fps = self.n_fps
        left_end_pos = abs_pos[3*n_fps : 3*n_fps + 3]
        left_end_quat = abs_pos[3*n_fps + 3 : 3*n_fps +7]
        right_end_pos = abs_pos[3*n_fps + 7 : 3*n_fps +10]
        right_end_quat = abs_pos[3*n_fps +10 : 3*n_fps +14]
        
        # Compute ralative normalized positions of feature points
        fps_pos = abs_pos[0:3*n_fps].reshape((3,n_fps))
        fps_pos_rel = cs.DM.zeros(fps_pos.shape)
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



def main():
    pass


