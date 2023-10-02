import torch
import numpy as np
import casadi as cs

import torch_rbf
import casadi_rbf


N_TESTS = 10


def _create_torch_rbf(in_features, out_features):
    basis_fcn = torch_rbf.gaussian
    rbf = torch_rbf.RBF(
        in_features, 
        out_features, 
        basis_fcn, 
        bTrainMuBeta=False
    )
    return rbf


def _create_casadi_rbf(centers, inv_sigmas):
    basis_fcn = casadi_rbf.gaussian
    rbf = casadi_rbf.RBF(
        centers,
        inv_sigmas,
        basis_fcn
    )
    return rbf


def test_RBF():
    in_features = [4, 10, 50, 100, 300]
    out_features = [2, 5, 25, 85, 256]

    for n_in, n_out in zip(in_features, out_features):
        rbf_torch = _create_torch_rbf(n_in, n_out)
        centers = np.array(rbf_torch.centres)
        inv_sigmas = np.array(rbf_torch.sigmas)

        rbf_casadi = _create_casadi_rbf(cs.DM(centers), cs.DM(inv_sigmas))

        for _ in range(N_TESTS):
            x_torch = torch.rand(1, n_in)
            x_np = np.array(x_torch)
            
            y_torch = rbf_torch.forward(x_torch)
            y_casadi = rbf_casadi(x_np)
            
            np.testing.assert_array_almost_equal(
                np.array(y_torch).ravel(),
                np.array(y_casadi).ravel()
            )


if __name__ == '__main__':
    test_RBF()
