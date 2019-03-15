

"""
 Stability analysis of the
 D1Q222 solver for the Euler system

 d_t(rho)   + d_x(rho u)     = 0,
 d_t(rho u) + d_x(rho u^2+p) = 0,
 d_t(E)   + d_x((E+p) u)     = 0,

 where E and p are linked by E = 1/2 rho u^2 + p/(gamma-1)

 then p = (gamma-1)(E - rho u^2/2)
 rho u^2 + p = (gamma-1)E + rho u^2 (3-gamma)/2
 E + p = 1/2 rho u^2 + p (1)

"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
RHO, Q, E, X = sp.symbols('rho, q, E, X')

# symbolic parameters
LA = sp.symbols('lambda', constants=True)
GAMMA = sp.Symbol('gamma', constants=True)
SIGMA_RHO, SIGMA_U, SIGMA_P = sp.symbols(
    'sigma_1, sigma_2, sigma_3',
    constants=True
)
symb_s_rho = 1/(.5+SIGMA_RHO)    # symbolic relaxation parameter
symb_s_u = 1/(.5+SIGMA_U)        # symbolic relaxation parameter
symb_s_p = 1/(.5+SIGMA_P)        # symbolic relaxation parameter

# numerical parameters
gamma = 1.4                      # gamma pressure law
la = 3.                          # velocity of the scheme
s_rho, s_u, s_p = 1.9, 1.5, 1.4  # relaxation parameters

dico = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': RHO,
            'polynomials': [1, X],
            'relaxation_parameters': [0, symb_s_rho],
            'equilibrium': [RHO, Q],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': Q,
            'polynomials': [1, X],
            'relaxation_parameters': [0, symb_s_u],
            'equilibrium': [Q, (GAMMA-1)*E+(3-GAMMA)/2*Q**2/RHO],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': E,
            'polynomials': [1, X],
            'relaxation_parameters': [0, symb_s_p],
            'equilibrium': [E, GAMMA*E*Q/RHO-(GAMMA-1)/2*Q**3/RHO**2],
        },
    ],
    'parameters': {
        LA: la,
        SIGMA_RHO: 1/s_rho-.5,
        SIGMA_U: 1/s_u-.5,
        SIGMA_P: 1/s_p-.5,
        GAMMA: gamma,
    },
}

scheme = pylbm.Scheme(dico, formal=True)
stab = pylbm.Stability(scheme)

# linearization around a state
rhoo = 1
uo = 0.5
po = 1
qo = rhoo * uo
Eo = .5*rhoo*uo**2 + po/(gamma-1.)

stab.visualize(
    {
        'linearization': {
            RHO: rhoo,
            Q: qo,
            E: Eo,
        },
        'number_of_wave_vectors': 1024,
    }
)
