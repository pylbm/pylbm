

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a nine velocities scheme for Navier-Stokes equations
"""
import sympy as sp
import pylbm

# pylint: disable=invalid-name

X, Y = sp.symbols('X, Y')
rho, qx, qy = sp.symbols('rho, qx, qy')
LA, C, SIGMA_MU, SIGMA_ETA = sp.symbols('lambda, c, mu, eta', constants=True)

s3 = 1/(0.5+SIGMA_MU)
s4 = s3
s5 = s4
s6 = s4
s7 = 1/(0.5+SIGMA_ETA)
s8 = s7
s = [0, 0, 0, s3, s4, s5, s6, s7, s8]

qx2 = qx**2
qy2 = qy**2
q2 = qx2+qy2
qxy = qx*qy
ux = qx/rho   # fluid velocity in the x-direction
uy = qy/rho   # fluid velocity in the y-direction
c2 = C*C  # square of the sound speed

scheme_cfg = {
    'parameters': {
        LA: 1.,
        SIGMA_MU: 1.e-2,
        SIGMA_ETA: 1.e-2,
    },
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': list(range(9)),
            'conserved_moments': [rho, qx, qy],
            'polynomials':[
                1, X, Y,
                X**2 + Y**2, X**2 - Y**2, X*Y,
                X*Y**2, Y*X**2,
                X**2*Y**2
            ],
            'relaxation_parameters': s,
            'equilibrium': [
                rho, qx, qy,
                2*c2*rho + rho*(ux**2+uy**2),
                rho*(ux**2-uy**2), rho*ux*uy,
                c2*qx + rho*ux*uy**2,
                c2*qy + rho*ux**2*uy,
                rho*(c2+ux**2)*(c2+uy**2)
            ],
        },
    ],
}

scheme = pylbm.Scheme(scheme_cfg)
eq_pde = pylbm.EquivalentEquation(scheme)

print(eq_pde)
