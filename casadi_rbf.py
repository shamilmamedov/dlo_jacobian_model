import casadi as cs
import numpy as np


class RBF:
    # NOTE You can compute each output distances separately in parlllel

    def __init__(self, centers: cs.DM, inv_sigmas: cs.DM, basis_fcn):
        self.centers = centers
        self.inv_sigmas = inv_sigmas
        self.basis_func = basis_fcn
        self.out_features, self.in_features = centers.size()

        self.rbf_fcn = self._get_rbf_fcn()

    def _get_symbolic_rbf_expression(self, x: cs.MX = None):
        if x is None: x = cs.MX.sym('x', self.in_features, 1)

        # Initialize an array to store the distances
        dinstances = [cs.norm_2(x - self.centers[i, :].T) 
                      for i in range(self.out_features)]

        # Create a column vector of distances
        distances = cs.vertcat(*dinstances)

        scaled_distances = distances * self.inv_sigmas
        rbf_expr = self.basis_func(scaled_distances)
        return x, rbf_expr
    
    def _get_rbf_fcn(self):
        x, rbf = self._get_symbolic_rbf_expression()
        rbf_fcn = cs.Function('rbf', [x], [rbf])
        # rbf_fcn = cs.Function('rbf', [x], [rbf], {'jit': True})
        return rbf_fcn
    
    def __call__(self, x):
        return self.rbf_fcn(x)
        # # Initialize an array to store the distances
        # dinstances = [cs.norm_2(x.reshape((-1, 1)) - self.centers[i, :].T) 
        #               for i in range(self.out_features)]

        # # Create a column vector of distances
        # distances = cs.vertcat(*dinstances)

        # scaled_distances = distances * self.inv_sigmas
        # rbf = self.basis_func(scaled_distances)
        # return rbf


def gaussian(alpha):
    phi = cs.exp(-1*cs.power(alpha, 2))
    return phi


def gaussian_jacobian(alpha):
    phi = cs.exp(-1*cs.power(alpha, 2))
    dphi = -2*alpha*phi
    return dphi



if __name__ == '__main__':
    in_features = 4
    out_features = 2

    basis_fcn = gaussian

    rbf = RBF(in_features, out_features, basis_fcn)
    rbf._get_symbolic_rbf_expression()