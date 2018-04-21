from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a two velocities scheme for the advection equation in 1D
"""
import sympy as sp
import pylbm
u, X = sp.symbols('u,X')
LA, c, sigma = sp.symbols('LA, c, sigma')

d = {
    'dim':1,
    'scheme_velocity':LA,
    'schemes':[
        {
            'velocities': [1,2],
            'conserved_moments':u,
            'polynomials': [1, X],
            'equilibrium': [u, u**2/2],
            'relaxation_parameters': [0, 1/(sigma+sp.Rational(1,2))],
        },
    ],
    'parameters':{LA:1., c:.1, sigma:1./1.9-.5},
    'consistency':{
        'order':2,
        'linearization':{u:c}
    }
}
s = pylbm.Scheme(d)
