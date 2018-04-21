from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D1Q2 for advection
"""
import sympy as sp
import pylbm
u, X = sp.symbols('u, X')

d = {
    'dim':1,
    'scheme_velocity':1.,
    'schemes':[
        {
            'velocities': [1, 2],
            'conserved_moments':u,
            'polynomials': [1, X],
            'equilibrium': [u, .5*u],
            'relaxation_parameters': [0., 1.9],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
