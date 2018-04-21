from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 4 velocities scheme for the advection equation in 2D
"""
import sympy as sp
import pylbm
u, X, Y = sp.symbols('u,X,Y')
cx, cy = .1, .0
d = {
    'dim':2,
    'scheme_velocity':1.,
    'schemes':[
        {
            'velocities': [1, 2, 3, 4],
            'conserved_moments':u,
            'polynomials': [1, X, Y, X**2-Y**2],
            'equilibrium': [u, cx*u, cy*u, 0.],
            'relaxation_parameters': [0., 1.9, 1.9, 1.2],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
print(s.generator.code)
