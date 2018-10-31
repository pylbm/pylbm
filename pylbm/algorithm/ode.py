
# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sympy as sp
from sympy import Eq

def euler(lhs, rhs):
    dt = sp.Symbol('dt')
    return Eq(lhs, lhs + dt/2*rhs)