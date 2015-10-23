# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from six.moves import range
import sympy as sp
from sympy.matrices import Matrix, zeros

from .stencil import Stencil
from .scheme import Scheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in range(25)] for i in range(10)]

def D1Q3(la=1., alpha=0.5, relaxparam=1.9):
    dim = 1 # spatial dimension
    n = 1 # number of elementary schemes
    V = ([2,0,1],) # list of velocities for each elementary scheme
    P = (Matrix([1,la*X,X**2/2]),) # polynomials that define the moments
    EQ = (Matrix([u[0][0], u[0][1], (alpha*la)**2/2*u[0][0]]),)
    s = ([0,0,relaxparam],)
    return Scheme(dim, Stencil(dim,V), P, EQ, s, la)

def D2Q9(la=1., rhoo=1., relaxparam=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5]):
    dim = 2 # spatial dimension
    n = 1 # number of elementary schemes
    # Velocities and polynoms for each elementary scheme
    V1 = [0,1,2,3,4,5,6,7,8]
    P1 = Matrix([1, la*X, la*Y, 3*(X**2+Y**2)-4, (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2, 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y, X**2-Y**2, X*Y])
    V = (V1, )
    P = (P1, )
    # Equilibria for each elementary scheme
    rhoo = 1.
    dummy = 1./(la**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]
    EQ1 = Matrix([u[0][0], u[0][1], u[0][3], -2*u[0][0] + 3*q2, u[0][0]+1.5*q2, u[0][1]/la, u[0][2]/la, qx2-qy2, qxy])
    s1 = [0, 0, 0, relaxparam[0], relaxparam[1], relaxparam[2], relaxparam[3], relaxparam[4], relaxparam[5]]
    EQ = (EQ1, )
    s = (s1, )
    return Scheme(dim, Stencil(dim,V), P, EQ, s, la)

if __name__ == "__main__":
    scheme = D2Q9()
    print(scheme.Code_Transport)
    print(scheme.Code_Equilibrium)
    print(scheme.Code_Relaxation)
    print(scheme.Code_m2F)
    print(scheme.Code_F2m)
