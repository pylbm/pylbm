from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D1Q3 for the wave equation
"""
import sympy as sp
import pylbm
u, v, X = sp.symbols('u, v, X')

c = 0.5
d = {
  'dim':1,
  'scheme_velocity':1.,
  'schemes':[{
    'velocities': [0, 1, 2],
    'conserved_moments':[u, v],
    'polynomials': [1, X, 0.5*X**2],
    'equilibrium': [u, v, .5*c**2*u],
    'relaxation_parameters': [0., 0., 1.9],
    },
  ],
}
s = pylbm.Scheme(d)
print(s)
