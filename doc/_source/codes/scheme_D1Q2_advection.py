# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D1Q2 for advection
"""
import sympy as sp
import pylbm

u, X = sp.symbols("u, X")

d = {
    "dim": 1,
    "scheme_velocity": 1.0,
    "schemes": [
        {
            "velocities": [1, 2],
            "conserved_moments": u,
            "polynomials": [1, X],
            "equilibrium": [u, 0.5 * u],
            "relaxation_parameters": [0.0, 1.9],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
