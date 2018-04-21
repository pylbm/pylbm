from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a vectorial two velocities scheme for the wave equation in 1D
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
            'velocities': [1,2],
            'conserved_moments':u,
            'polynomials': [1, X],
            'equilibrium': [u, v],
            'relaxation_parameters': [0, 1/(sigma+sp.Rational(1,2))],
        },
        {
            'velocities': [1,2],
            'conserved_moments':v,
            'polynomials': [1, X],
            'equilibrium': [v, c**2*u],
            'relaxation_parameters': [0, 1/(sigma+sp.Rational(1,2))],
        },
    ],
    'parameters':{LA:1., c:.1, sigma:1./1.9-.5},
    'consistency':{
        'order':2,
        'numeric':True,
    }
}
s = pylbm.Scheme(d)
