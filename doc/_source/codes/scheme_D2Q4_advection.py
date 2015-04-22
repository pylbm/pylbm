import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM
X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

d = {
  'dim':2,
  'scheme_velocity':1.,
  'schemes':[{
    'velocities': range(1,5),
    'polynomials': Matrix([1, X, Y, X**2-Y**2]),
    'equilibrium': Matrix([u[0][0], .1*u[0][0], .2*u[0][0], 0.]),
    'relaxation_parameters': [0., 1.9, 1.9, 1.4],
    },
  ],
}
s = pyLBM.Scheme(d)
print(s)
