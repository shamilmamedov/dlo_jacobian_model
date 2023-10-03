import casadi as cs
import time
import numpy as np

import casadi_nn


def linear_fcn(W, x):
    return W @ x


def test_nn_custom_jacobian():

    W = cs.DM.rand(256, 256)
    x = cs.MX.sym('x', 256, 1)
    p_dummy = cs.MX.sym('p_dummy', 256, 1)
    f_expr = casadi_nn.Linear(W)(x)

    jac_autodiff = cs.Function("jac_autodiff", [x], [cs.jacobian(f_expr, x)])
    print(jac_autodiff.n_instructions())

    custom_jacobian = cs.Function("jac_f", [x, p_dummy], [W], ['x', 'dummy'], ['jac_f_x'])
    print(custom_jacobian.n_instructions())
    f = cs.Function("f", [x], [f_expr], ['x'], ['f'],
                    dict(custom_jacobian = custom_jacobian, jac_penalty = 0, always_inline=False, never_inline=True))

    jac_custom = cs.Function('JJ', [x], [f.jacobian()(x, x)])
    print(jac_custom.n_instructions())

    x = cs.DM.rand(256, 1)
    avg_time = []
    n_iter = 1000
    for k in range(n_iter):
        start = time.time()
        jac_autodiff(x)
        end = time.time()
        avg_time.append((end-start))
    print(f'Average autodiff time: {1000*np.mean(avg_time):.3f}ms')

    avg_time = []
    n_iter = 1000
    for k in range(n_iter):
        start = time.time()
        jac_custom(x)
        end = time.time()
        avg_time.append((end-start))
    print(f'Average custom time: {1000*np.mean(avg_time):.3f}ms')


if __name__ == '__main__':
    test_nn_custom_jacobian()