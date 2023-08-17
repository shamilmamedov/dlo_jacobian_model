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
        theta = self.rbf(x, rbf_centers, rbf_sigmas)
        out = self.linear(theta, lin_A, lin_b)
        # return out.reshape((3*self.n_fps, 12))
        # out = out.reshape((3*12,))
        return out



def main():
    pass


