from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a nine velocities scheme for Navier-Stokes equations
"""
import sympy as sp
import pylbm

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')
rhoo, ux, uy = sp.symbols('rhoo, ux, uy')
sigma_mu, sigma_eta = sp.symbols('sigma_mu, sigma_eta')

rhoo_num = 1.
la = 1.
s3 = 1/(sigma_mu+sp.Rational(1, 2))
s4 = s3
s5 = s4
s6 = s4
s7 = 1/(sigma_eta+sp.Rational(1, 2))
s8 = s7
s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]

dummy = 1/(LA**2*rhoo)
qx2 = dummy*qx**2
qy2 = dummy*qy**2
q2  = qx2+qy2
qxy = dummy*qx*qy

dico = {
    'parameters':{LA:la, rhoo:rhoo_num, sigma_mu:1.e-2, sigma_eta:1.e-2},
    'dim':2,
    'scheme_velocity':LA,
    'schemes':[
        {
        'velocities':range(9),
        'conserved_moments':[rho, qx, qy],
        'polynomials':[
            1,
            LA*X, LA*Y,
            3*(X**2+Y**2)-4,
            (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2,
            3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
            X**2-Y**2, X*Y
        ],
        'relaxation_parameters':s,
        'equilibrium':[rho, qx, qy,
            -2*rho + 3*q2,
            rho - 3*q2,
            -qx/LA, -qy/LA,
            qx2-qy2, qxy
        ],
        },
    ],
    'consistency':{
        'order':2,
        'linearization':{rho: 1, qx: 0, qy: 0},
        #'linearization':{rho: rhoo, qx: rhoo*ux, qy: rhoo*uy},
    },
}

S = pylbm.Scheme(dico)
