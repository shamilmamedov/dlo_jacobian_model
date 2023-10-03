import casadi as cs
import time
import numpy as np

import casadi_nn
import casadi_rbf


def time_execution(fcn, args, n_iter=1000, fcn_name=''):
    avg_time = []
    for k in range(n_iter):
        start = time.time()
        fcn(args)
        end = time.time()
        avg_time.append((end-start))
    print(f'Average {fcn_name} evaluation time: {1000*np.mean(avg_time):.3f}ms')


def test_nn_custom_jacobian():
    print('Testing linear layer')
    W = cs.DM.rand(256, 256)
    x = cs.MX.sym('x', 256, 1)
    model = casadi_nn.Linear(W)

    f_expr = model(x)
    jac_autodiff = cs.Function("jac_autodiff", [x], [cs.jacobian(f_expr, x)])
    print(jac_autodiff.n_instructions())

    jac_custom = cs.Function('JJ', [x], [model.linear_fcn.jacobian()(x, x)])
    print(jac_custom.n_instructions())

    n_iter = 1000
    x = cs.DM.rand(256, 1)
    time_execution(jac_autodiff, x, n_iter, fcn_name='autodiff')
    time_execution(jac_custom, x, n_iter, fcn_name='custom')


def test_gaussian():
    print('\nTesting gaussian')
    x = cs.MX.sym('x')
    f_expr = casadi_rbf.gaussian(x)

    jac_autodiff = cs.Function("jac_autodiff", [x], [cs.jacobian(f_expr, x)])
    print(jac_autodiff.n_instructions())

    jac_custom = cs.Function('jac_custom', [x], [casadi_rbf.gaussian_jacobian(x)])
    print(jac_custom.n_instructions())

    # for _ in range(10):
    #     x = cs.DM.rand()
    #     assert jac_autodiff(x) == jac_custom(x)

    n_iter = 1000
    x = cs.DM.rand(256,1)
    time_execution(jac_autodiff, x, n_iter, fcn_name='autodiff')
    time_execution(jac_custom, x, n_iter, fcn_name='custom')
    


if __name__ == '__main__':
    test_nn_custom_jacobian()
    test_gaussian()