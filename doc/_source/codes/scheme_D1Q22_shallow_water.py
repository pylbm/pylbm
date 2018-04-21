from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a double D1Q2 for shallow water
"""
import sympy as sp
import pylbm

# parameters
h, q, X, LA, g = sp.symbols('h, q, X, LA, g')
la = 2.              # velocity of the scheme
s_h, s_q = 1.7, 1.5  # relaxation parameters

d = {
    'dim': 1,
    'scheme_velocity': la,
    'schemes':[
        {
            'velocities': [1, 2],
            'conserved_moments': h,
            'polynomials': [1, LA*X],
            'relaxation_parameters': [0, s_h],
            'equilibrium': [h, q],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': q,
            'polynomials': [1, LA*X],
            'relaxation_parameters': [0, s_q],
            'equilibrium': [q, q**2/h+.5*g*h**2],
        },
    ],
    'parameters': {LA: la, g: 1.},
}
s = pylbm.Scheme(d)
print(s)
