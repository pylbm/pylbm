"""
 Stability analysis of the
 D1Q222 solver for the Euler system

 d_t(rho)   + d_x(rho u)     = 0,
 d_t(rho u) + d_x(rho u^2+p) = 0,
 d_t(E)   + d_x((E+p) u)     = 0,

 where E and p are linked by E = 1/2 rho u^2 + p/(gamma-1)

 then p = (gamma-1)(E - rho u^2/2)
 rho u^2 + p = (gamma-1)E + rho u^2 (3-gamma)/2
"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
RHO, Q, E, X = sp.symbols('rho, q, E, X')

# symbolic parameters
LA = sp.symbols('lambda', constants=True)
GAMMA = sp.Symbol('gamma', constants=True)
S_RHO, S_U, S_P = sp.symbols('s_1, s_2, s_3', constants=True)

# numerical parameters
gamma = 1.4                      # gamma pressure law
la = 3.                          # velocity of the scheme
s_rho, s_u, s_p = 1.9, 1.9, 1.9  # relaxation parameters

dico = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': RHO,
            'polynomials': [1, X],
            'relaxation_parameters': [0, S_RHO],
            'equilibrium': [RHO, Q],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': Q,
            'polynomials': [1, X],
            'relaxation_parameters': [0, S_U],
            'equilibrium': [Q, (GAMMA-1)*E+(3-GAMMA)/2*Q**2/RHO],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': E,
            'polynomials': [1, X],
            'relaxation_parameters': [0, S_P],
            'equilibrium': [E, GAMMA*E*Q/RHO-(GAMMA-1)/2*Q**3/RHO**2],
        },
    ],
    'parameters': {
        LA: la,
        S_RHO: s_rho,
        S_U: s_u,
        S_P: s_p,
        GAMMA: gamma,
    },
}

scheme = pylbm.Scheme(dico)
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
        'parameters': {
            LA: {
                'range': [1, 200],
                'init': la,
                'step': 1
            },
            RHO: {
                'range': [0, 20],
                'init': rhoo,
                'step': .1
            },
            Q: {
                'range': [0, 1],
                'init': qo,
                'step': .01
            },
            E: {
                'range': [0, 10],
                'init': Eo,
                'step': .1
            },
            S_RHO: {
                'name': r"$s_{\rho}$",
                'range': [0, 2],
                'init': s_rho,
                'step': .01
            },
            S_U: {
                'name': r"$s_{u}$",
                'range': [0, 2],
                'init': s_u,
                'step': .01
            },
            S_P: {
                'name': r"$s_{p}$",
                'range': [0, 2],
                'init': s_p,
                'step': .01
            },
        },
        'number_of_wave_vectors': 1024,
    }
)
