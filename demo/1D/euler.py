from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2Q2 for the Euler system on [0, 1]

 d_t(rho)   + d_x(rho u)     = 0, t > 0, 0 < x < 1,
 d_t(rho u) + d_x(rho u^2+p) = 0, t > 0, 0 < x < 1,
 d_t(E)   + d_x((E+p) u)     = 0, t > 0, 0 < x < 1,

 where E and p are linked by E = 1/2 rho u^2 + p/(gamma-1)

 then p = (gamma-1)(E - rho u^2/2)
 rho u^2 + p = (gamma-1)E + rho u^2 (3-gamma)/2
 E + p = 1/2 rho u^2 + p (1)

 Initial and boundary conditions are:

 rho(t=0,x) = rho0(x), u(t=0,x) = u0(x), E(t=0,x) = E0(x)
 d_t(rho)(t,x=0) = d_t(rho)(t,x=1) = 0
 d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0
 d_t(E)(t,x=0) = d_t(E)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to simulate the Sod's shock tube

 test: True
"""

import sympy as sp
import numpy as np
import pylbm

rho, q, E, X, LA = sp.symbols('rho, q, E, X, LA')

def Riemann_pb(x, xmin, xmax, uL, uR):
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    u[x < xm] = uL
    u[x == xm] = .5*(uL+uR)
    u[x > xm] = uR
    return u

def run(dx, Tf, generator="cython", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    gamma = 1.4
    xmin, xmax = 0., 1.
    la = 3. # velocity of the scheme
    rho_L, rho_R, p_L, p_R, u_L, u_R = 1., 1./8., 1., 0.1, 0., 0.
    q_L = rho_L*u_L
    q_R = rho_R*u_R
    E_L = rho_L*u_L**2 + p_L/(gamma-1.)
    E_R = rho_R*u_R**2 + p_R/(gamma-1.)
    s_rho, s_q, s_E = 1.9, 1.5, 1.4

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':rho,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_rho],
                'equilibrium':[rho, q],
                'init':{rho:(Riemann_pb, (xmin, xmax, rho_L, rho_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':q,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_q],
                'equilibrium':[q, (gamma-1.)*E+0.5*(3.-gamma)*q**2/rho],
                'init':{q:(Riemann_pb, (xmin, xmax, q_L, q_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':E,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_E],
                'equilibrium':[E, gamma*E*q/rho-0.5*(gamma-1.)*q**3/rho**2],
                'init':{E:(Riemann_pb, (xmin, xmax, E_L, E_R))},
            },
        ],
        'boundary_conditions':{
            0:{
                'method':{
                    0: pylbm.bc.Neumann,
                    1: pylbm.bc.Neumann,
                    2: pylbm.bc.Neumann
                },
            },
        },
        'parameters':{LA:la},
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    while (sol.t<Tf):
        sol.one_time_step()

    if withPlot:
        x = sol.domain.x
        rho_n = sol.m[rho]
        q_n = sol.m[q]
        E_n = sol.m[E]
        u = q_n/rho_n
        p = (gamma-1.)*(E_n - .5*rho_n*u**2)
        e = E_n/rho_n - .5*u**2

        viewer= pylbm.viewer.matplotlibViewer
        fig = viewer.Fig(2, 3)

        fig[0,0].plot(x, rho_n)
        fig[0,0].title = 'mass'
        fig[0,1].plot(x, u)
        fig[0,1].title = 'velocity'
        fig[0,2].plot(x, p)
        fig[0,2].title = 'pressure'
        fig[1,0].plot(x, E_n)
        fig[1,0].title = 'energy'
        fig[1,1].plot(x, q_n)
        fig[1,1].title = 'momentum'
        fig[1,2].plot(x, e)
        fig[1,2].title = 'internal energy'

        fig.show()

    return sol

if __name__ == '__main__':
    dx = 1.e-3
    Tf = .14
    run(dx, Tf, generator='numpy')
