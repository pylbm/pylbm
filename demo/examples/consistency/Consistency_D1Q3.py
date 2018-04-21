from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a three velocities scheme for the wave equation in 1D
"""
import sympy as sp
import pylbm
u, v, X = sp.symbols('u, v, X')
LA, c, sigma = sp.symbols('LA, c, sigma')

d = {
    'dim':1,
    'scheme_velocity':LA,
    'schemes':[
        {
            'velocities': range(3),
            'conserved_moments':[u, v],
            'polynomials': [1, LA*X, (LA*X)**2],
            'equilibrium': [u, v, c**2*u],
            'relaxation_parameters': [0, 0, 1/(sigma+sp.Rational(1,2))],
        },
    ],
    'parameters':{LA:1., c:.1, sigma:1./1.9-.5},
    'consistency':{'order':2}
}
s = pylbm.Scheme(d)
