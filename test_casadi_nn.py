import torch
import numpy as np

import casadi_nn

def test_linear():
    in_features = 5
    out_features = 2
    bias = False

    linear_torch = torch.nn.Linear(in_features, out_features, bias)
    linear_casadi = casadi_nn.Linear(in_features, out_features, bias)

    A = linear_torch.weight.detach().numpy()
    
    x_torch = torch.rand(1, in_features)
    x_np = np.array(x_torch)
    
    y_torch = linear_torch.forward(x_torch)
    y_casadi = linear_casadi(x_np, A)
    np.testing.assert_array_almost_equal(
        y_torch.detach().numpy().ravel(),
        np.array(y_casadi).ravel()
    )

if __name__ == '__main__':
    test_linear()