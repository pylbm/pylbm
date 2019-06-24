

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a three velocities scheme for the wave equation in 1D
"""
import sympy as sp
import pylbm

# pylint: disable=invalid-name

U, V, X = sp.symbols('u, v, X')
LA, C, SIGMA = sp.symbols('lambda, c, sigma', constants=True)

scheme_cfg = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': list(range(3)),
            'conserved_moments': [U, V],
            'polynomials': [1, X, X**2],
            'equilibrium': [U, V, C**2*U],
            'relaxation_parameters': [0, 0, 1/(0.5+SIGMA)],
        },
    ],
    'parameters': {
        LA: 1.,
        C: 0.1,
        SIGMA: 1./1.9-.5,
    },
}

scheme = pylbm.Scheme(scheme_cfg)
eq_pde = pylbm.EquivalentEquation(scheme)

print(eq_pde)
