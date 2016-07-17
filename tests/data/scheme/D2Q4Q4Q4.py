from six.moves import range
import numpy as np
import sympy as sp
from sympy import Matrix

X, Y, Z, LA, g = sp.symbols('X,Y,Z,LA,g')
rho, qx, qy = sp.symbols('rho, qx, qy')
m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in range(25)] for i in range(10)]

velocity = list(range(1, 5))

relax = [0]*4

poly = [[1, LA*X, LA*Y, X**2-Y**2],
            ['1', 'LA*X', 'LA*Y', 'X**2-Y**2']]

eq = [[[m[0][0], m[1][0], m[2][0], 0.],
       [m[1][0], m[1][0]**2/m[0][0] + 0.5*g*m[0][0]**2, m[1][0]*m[2][0]/m[0][0], 0.],
       [m[2][0], m[1][0]*m[2][0]/m[0][0], m[2][0]**2/m[0][0] + 0.5*g*m[0][0]**2, 0.]],
       [[rho, qx, qy, 0.],
       [qx, qx**2/rho + .5*g*rho**2, qx*qy/rho, 0.],
       [qy, qx*qy/rho, qy**2/rho + 0.5*g*rho**2, 0.]]
       ]

param = [{LA: 1., g: 1.}, {'LA': 1., 'g': 1.}]

consm = [[None, None, None], [rho, qx, qy]]

EQ_result = [Matrix([[m[0][0]],
                     [m[1][0]],
                     [m[2][0]],
                     [    0.0]]),
            Matrix([[                              m[1][0]],
                    [0.5*g*m[0][0]**2 + m[1][0]**2/m[0][0]],
                    [              m[1][0]*m[2][0]/m[0][0]],
                    [                                  0.0]]),
            Matrix([[                              m[2][0]],
                    [              m[1][0]*m[2][0]/m[0][0]],
                    [0.5*g*m[0][0]**2 + m[2][0]**2/m[0][0]],
                    [                                  0.0]])]

Mnum = [np.array([[ 1.,  1.,  1.,  1.],
       [ 1.,  0., -1.,  0.],
       [ 0.,  1.,  0., -1.],
       [ 1., -1.,  1., -1.]]), np.array([[ 1.,  1.,  1.,  1.],
       [ 1.,  0., -1.,  0.],
       [ 0.,  1.,  0., -1.],
       [ 1., -1.,  1., -1.]]), np.array([[ 1.,  1.,  1.,  1.],
       [ 1.,  0., -1.,  0.],
       [ 0.,  1.,  0., -1.],
       [ 1., -1.,  1., -1.]])]
