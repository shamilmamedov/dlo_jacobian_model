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
        # self.linear_fcn = self._get_linear_fcn()

    def _get_symbolic_expression(self, x: cs.SX = None):
        if x is None: x = cs.SX.sym('x', self.in_features, 1)
        out = self.W @ x
        
        if self.bias is not None:
            out = out + self.b
        return x, out 

    def _get_linear_fcn(self):
        x, out = self._get_symbolic_expression()
        # linear_fcn = cs.Function('linear', [x], [out])
        linear_fcn = cs.Function('linear', [x], [out])
        return linear_fcn
    
    def __call__(self, x):
        # return self.linear_fcn(x)
        out = self.W @ x.reshape((-1, 1))
        
        if self.bias is not None:
            out = out + self.b
        return out