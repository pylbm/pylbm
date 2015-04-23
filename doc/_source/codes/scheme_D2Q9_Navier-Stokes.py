import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM
X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

dx = 1./256    # space step
eta = 1.25e-5  # shear viscosity
kappa = 10*eta # bulk viscosity
sb = 1./(.5+kappa*3./dx)
ss = 1./(.5+eta*3./dx)
d = {
    'dim':2,
    'scheme_velocity':1.,
    'schemes':[{
        'velocities':range(9),
        'polynomials':Matrix([
            1, X, Y,
            3*(X**2+Y**2)-4,
            (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2,
            3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
            X**2-Y**2, X*Y
        ]),
        'relaxation_parameters':[0., 0., 0., sb, sb, sb, sb, ss, ss],
        'equilibrium':Matrix([
            u[0][0], u[0][1], u[0][2],
            -2*u[0][0] + 3*u[0][1]**2 + 3*u[0][2]**2,
            u[0][0] + 3/2*u[0][1]**2 + 3/2*u[0][2]**2,
            -u[0][1], -u[0][2],
            u[0][1]**2 - u[0][2]**2, u[0][1]*u[0][2]
        ]),
    },],
}
s = pyLBM.Scheme(d)
print(s)
