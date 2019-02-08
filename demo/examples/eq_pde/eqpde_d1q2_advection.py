

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a two velocities scheme for the advection equation in 1D
"""
import sympy as sp
import pylbm

# pylint: disable=invalid-name

U, X = sp.symbols('u, X')
LA, C, SIGMA = sp.symbols('lambda, c, sigma', constants=True)

scheme_cfg = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': U,
            'polynomials': [1, X],
            'equilibrium': [U, C*U],
            'relaxation_parameters': [0, 1/(0.5+SIGMA)],
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
