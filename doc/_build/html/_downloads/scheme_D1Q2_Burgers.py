import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM
X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

d = {
  'dim':1,
  'scheme_velocity':1.,
  'schemes':[{
    'velocities': range(1,3),
    'polynomials': Matrix([1, X]),
    'equilibrium': Matrix([u[0][0], .5*u[0][0]**2]),
    'relaxation_parameters': [0., 1.9],
    },
  ],
}
s = pyLBM.Scheme(d)
print(s)
