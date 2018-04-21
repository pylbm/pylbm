from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D2Q9 for Navier-Stokes
"""
from six.moves import range
import sympy as sp
import pylbm
rho, qx, qy, X, Y = sp.symbols('rho, qx, qy, X, Y')

dx = 1./256    # space step
eta = 1.25e-5  # shear viscosity
kappa = 10*eta # bulk viscosity
sb = 1./(.5+kappa*3./dx)
ss = 1./(.5+eta*3./dx)
d = {
    'dim':2,
    'scheme_velocity':1.,
    'schemes':[{
        'velocities':list(range(9)),
        'conserved_moments':[rho, qx, qy],
        'polynomials':[
            1, X, Y,
            3*(X**2+Y**2)-4,
            (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2,
            3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
            X**2-Y**2, X*Y
        ],
        'relaxation_parameters':[0., 0., 0., sb, sb, sb, sb, ss, ss],
        'equilibrium':[
            rho, qx, qy,
            -2*rho + 3*qx**2 + 3*qy**2,
            rho + 3/2*qx**2 + 3/2*qy**2,
            -qx, -qy,
            qx**2 - qy**2, qx*qy
        ],
    },],
}
s = pylbm.Scheme(d)
print(s)
