import torch
import numpy as np
import casadi as cs

import casadi_nn

N_TESTS = 50

def test_linear():
    in_features = [5, 10, 20, 50, 100, 256, 512]
    out_features = [2, 5, 15, 30, 120, 360, 10]
    bias = False

    for n_in, n_out in zip(in_features, out_features):
        linear_torch = torch.nn.Linear(n_in, n_out, bias)
        W = linear_torch.weight.detach().numpy()

        linear_casadi = casadi_nn.Linear(cs.DM(W))

        for _ in range(N_TESTS):
            x_torch = torch.rand(1, n_in)
            x_np = np.array(x_torch)
            
            y_torch = linear_torch.forward(x_torch)
            y_casadi = linear_casadi(x_np)
            np.testing.assert_array_almost_equal(
                y_torch.detach().numpy().ravel(),
                np.array(y_casadi).ravel()
            )

if __name__ == '__main__':
    test_linear()