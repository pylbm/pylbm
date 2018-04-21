from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 6 velocities scheme for the advection equation in 3D
"""
from six.moves import range
import sympy as sp
import pylbm
u, X, Y, Z = sp.symbols('u,X,Y,Z')
cx, cy, cz = .1, .0, .0
d = {
    'dim':3,
    'scheme_velocity':1.,
    'schemes':[
        {
            'velocities': list(range(1,7)),
            'conserved_moments':u,
            'polynomials': [1, X, Y, Z, X**2-Y**2, X**2-Z**2],
            'equilibrium': [u, cx*u, cy*u, cz*u, 0., 0.],
            'relaxation_parameters': [0., 1.9, 1.9, 1.9, 1.2, 1.2],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
print(s.generator.code)
