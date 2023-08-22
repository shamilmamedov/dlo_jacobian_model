import casadi as cs
import numpy as np



class RBF:
    # NOTE You can compute each output distances separately in parlllel

    def __init__(self, in_features, out_features, basis_fcn):
        self.in_features = in_features
        self.out_features = out_features
        self.basis_func = basis_fcn

        self.rbf_fcn = self._get_rbf_fcn()

    def _get_symbolic_rbf_expression(self, x: cs.SX = None):
        centers = cs.SX.sym('c', self.out_features, self.in_features)
        inv_sigmas = cs.SX.sym('σ', self.out_features, 1)
        if x is None: x = cs.SX.sym('x', self.in_features, 1)

        dinstances = cs.SX.zeros(self.out_features,1)
        for i in range(self.out_features):
            dinstances[i] = cs.norm_2(x - centers[i,:].T)

        scaled_distances = dinstances * inv_sigmas
        rbf_expr = self.basis_func(scaled_distances)

        p = cs.vertcat(cs.vec(centers), inv_sigmas)
        return x, p, rbf_expr
    
    def _get_rbf_fcn(self):
        x, p, rbf = self._get_symbolic_rbf_expression()
        rbf_fcn = cs.Function('rbf', [x, p], [rbf])
        return rbf_fcn
    
    def __call__(self, x, centers, inv_sigmas):
        if isinstance(centers, np.ndarray):
            centers = cs.DM(centers)

        if isinstance(inv_sigmas, np.ndarray):
            inv_sigmas = cs.DM(inv_sigmas)

        p = cs.vertcat(cs.vec(centers), inv_sigmas)
        return self.rbf_fcn(x, p)


def gaussian(alpha):
    phi = cs.exp(-1*cs.power(alpha, 2))
    return phi


if __name__ == '__main__':
    in_features = 4
    out_features = 2

    basis_fcn = gaussian

    rbf = RBF(in_features, out_features, basis_fcn)
    rbf._get_symbolic_rbf_expression()