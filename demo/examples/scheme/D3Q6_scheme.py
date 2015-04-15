import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM
X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

ux, uy, uz = .1, .0, .0

d = {
    'dim':3,
    'scheme_velocity':1.,
    'schemes':[{
        'velocities': range(1,7),
        'polynomials': Matrix([1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2]),
        'equilibrium': Matrix([u[0][0], ux*u[0][0], uy*u[0][0], uz*u[0][0], 0., 0.]),
        'relaxation_parameters': [0., 1.9, 1.9, 1.9, 1.2, 1.2],
    },],
}
s = pyLBM.Scheme(d)
print s

print s.generator.code
