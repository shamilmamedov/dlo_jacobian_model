import casadi as cs
import numpy as np

class Linear:
    def __init__(self, in_features:int, out_features:int, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear_fcn = self._get_linear_fcn()

    def _get_symbolic_expression(self):
        x = cs.SX.sym('x', self.in_features, 1)
        A = cs.SX.sym('A', self.out_features, self.in_features)
        out = A @ x
        
        if self.bias:
            b = cs.SX.sym('b', self.out_features, 1)
            out = out + b
            p = cs.vertcat(cs.vec(A), b)
            
            return x, p, out

        else:
            p = cs.vec(A)
            return x, p, out 

    def _get_linear_fcn(self):
        x, p, out = self._get_symbolic_expression()
        linear_fcn = cs.Function('linear', [x, p], [out])
        return linear_fcn
    
    def __call__(self, x, A, b=None):
        if isinstance(A, np.ndarray):
            A = cs.DM(A)

        if b is None:
            p = cs.vec(A)
            return self.linear_fcn(x, p)
        else:
            if isinstance(b, np.ndarray):
                b = cs.DM(b)
            p = cs.vertcat(cs.vec(A), b)
            return self.linear_fcn(x, p)