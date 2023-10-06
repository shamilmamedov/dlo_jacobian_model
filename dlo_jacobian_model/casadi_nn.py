import casadi as cs
import numpy as np

class Linear:
    """Linear NN layer: y = Ax + b where b is the bias and optional"""

    def __init__(self, W: cs.DM, bias: cs.DM = None):
        out_f, in_f = W.shape

        self.W = W
        self.bias = bias
        self.in_features = in_f
        self.out_features = out_f

        self.linear_fcn = self._get_linear_fcn()

    def _get_symbolic_expr(self, x: cs.SX = None):
        if x is None: x = cs.MX.sym('x', self.in_features, 1)
        out = self.W @ x
        
        if self.bias is not None:
            out = out + self.b
        return x, out 

    def _get_linear_fcn(self):
        x, out = self._get_symbolic_expr()
        jac_f = self._get_custom_jacobian_fcn()
        custom_jac_dict = dict(
            custom_jacobian = jac_f, 
            jac_penalty = 0, 
            always_inline=False, 
            never_inline=True
        )

        linear_fcn = cs.Function('linear', [x], [out], custom_jac_dict)
        return linear_fcn
    
    def _get_custom_jacobian_fcn(self):
        x = cs.MX.sym('x', self.in_features, 1)
        dummy = cs.MX.sym('dummy', self.out_features, 1)
        jac_f = cs.Function('jac_linear', [x, dummy], [self.W], 
                            ['x', 'dummy'], ['jac_f_x'])
        return jac_f

    def __call__(self, x):
        return self.linear_fcn(x)
