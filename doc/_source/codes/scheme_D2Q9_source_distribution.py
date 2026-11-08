# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D2Q9 scheme for the Navier-Stokes equations with a body force
added to the distribution functions after the relaxation
"""

import sympy as sp
import pylbm

rho, qx, qy, X, Y = sp.symbols("rho, qx, qy, X, Y")

fx, fy = 1.0e-5, 0.0  # body force

# isotropic weights of the D2Q9 stencil
stencil = pylbm.Stencil({"dim": 2, "schemes": [{"velocities": list(range(9))}]})
weights = {0: sp.Rational(4, 9), 1: sp.Rational(1, 9), 2: sp.Rational(1, 36)}
source = [
    3 * weights[abs(v.vx) + abs(v.vy)] * (v.vx * fx + v.vy * fy) for v in stencil.v[0]
]

d = {
    "dim": 2,
    "scheme_velocity": 1.0,
    "schemes": [
        {
            "velocities": list(range(9)),
            "conserved_moments": [rho, qx, qy],
            "polynomials": [
                1,
                X,
                Y,
                3 * (X**2 + Y**2) - 4,
                (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8) / 2,
                3 * X * (X**2 + Y**2) - 5 * X,
                3 * Y * (X**2 + Y**2) - 5 * Y,
                X**2 - Y**2,
                X * Y,
            ],
            "relaxation_parameters": [0.0, 0.0, 0.0, 1.1, 1.1, 1.7, 1.7, 1.5, 1.5],
            "equilibrium": [
                rho,
                qx,
                qy,
                -2 * rho + 3 * (qx**2 + qy**2),
                rho - 3 * (qx**2 + qy**2),
                -qx,
                -qy,
                qx**2 - qy**2,
                qx * qy,
            ],
            "source_distribution": source,
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
