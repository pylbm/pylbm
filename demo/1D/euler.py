
# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

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

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

RHO, Q, E, X = sp.symbols('rho, q, E, X')
LA = sp.symbols('lambda', constants=True)


def riemann_pb(x, xmin, xmax, u_left, u_right):
    """
    initial condition for a Riemann problem

    Parameters
    ----------

    x: ndarray
        the space vectoe
    xmin: float
        left bound of the domain
    xmax: float
        right bound of the domain
    u_left: float
        left value
    u_right: float
        right value

    Returns
    -------

    ndarray
        the values of the Riemann problem
    """
    x_middle = 0.5*(xmin+xmax)
    out = np.empty(x.shape)
    out[x < x_middle] = u_left
    out[x == x_middle] = .5*(u_left+u_right)
    out[x > x_middle] = u_right
    return out


def run(space_step,
        final_time,
        generator="cython",
        sorder=None,
        with_plot=True):
    """
    Parameters
    ----------

    space_step: double
        spatial step

    final_time: double
        final time

    generator: string, optional
        pylbm generator

    sorder: list, optional
        storage order

    with_plot: boolean, optional
        if True plot the solution otherwise just compute the solution

    Returns
    -------

    sol
        <class 'pylbm.simulation.Simulation'>

    """
    # parameters
    gamma = 1.4                         # ratio of specific heats
    xmin, xmax = 0., 1.                 # bounds of the domain
    la = 3.                             # lattice velocity (la=dx/dt)
    s_rho, s_q, s_rhoe = 1.9, 1.5, 1.4  # relaxation parameters

    rho_left, p_left, u_left = 1, 1, 0         # left state
    rho_right, p_right, u_right = 1/8, 0.1, 0  # right state
    q_left = rho_left*u_left
    q_right = rho_right*u_right
    rhoe_left = rho_left*u_left**2 + p_left/(gamma-1.)
    rhoe_right = rho_right*u_right**2 + p_right/(gamma-1.)

    simu_cfg = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': space_step,
        'lattice_velocity': LA,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': RHO,
                'polynomials': [1, X],
                'relaxation_parameters': [0, s_rho],
                'equilibrium': [RHO, Q],
            },
            {
                'velocities': [1, 2],
                'conserved_moments': Q,
                'polynomials': [1, X],
                'relaxation_parameters': [0, s_q],
                'equilibrium': [Q, (gamma-1)*E+(3-gamma)/2*Q**2/RHO],
            },
            {
                'velocities': [1, 2],
                'conserved_moments': E,
                'polynomials': [1, X],
                'relaxation_parameters': [0, s_rhoe],
                'equilibrium': [E, gamma*E*Q/RHO-(gamma-1)/2*Q**3/RHO**2],
            },
        ],
        'init': {
            RHO: (riemann_pb, (xmin, xmax, rho_left, rho_right)),
            Q: (riemann_pb, (xmin, xmax, q_left, q_right)),
            E: (riemann_pb, (xmin, xmax, rhoe_left, rhoe_right))
        },
        'boundary_conditions': {
            0: {
                'method': {
                    0: pylbm.bc.Neumann,
                    1: pylbm.bc.Neumann,
                    2: pylbm.bc.Neumann
                },
            },
        },
        'parameters': {LA: la},
        'generator': generator,
    }

    # build the simulation
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    with pylbm.progress_bar(int(final_time/sol.dt), title='run') as pbar:
        while sol.t < final_time:
            sol.one_time_step()
            pbar()

    if with_plot:
        x = sol.domain.x
        rho_n = sol.m[RHO]
        q_n = sol.m[Q]
        rhoe_n = sol.m[E]
        u_n = q_n/rho_n
        p_n = (gamma-1.)*(rhoe_n - .5*rho_n*u_n**2)
        e_n = rhoe_n/rho_n - .5*u_n**2

        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(2, 3)

        ax_rho = fig[0, 0]
        ax_v = fig[0, 1]
        ax_p = fig[0, 2]
        ax_rhoe = fig[1, 0]
        ax_q = fig[1, 1]
        ax_e = fig[1, 2]
        ax_rho.plot(x, rho_n)
        ax_v.plot(x, u_n)
        ax_p.plot(x, p_n)
        ax_rhoe.plot(x, rhoe_n)
        ax_q.plot(x, q_n)
        ax_e.plot(x, e_n)
        ax_rho.title = 'mass'
        ax_v.title = 'velocity'
        ax_p.title = 'pressure'
        ax_rhoe.title = 'energy'
        ax_q.title = 'momentum'
        ax_e.title = 'internal energy'

        fig.show()

    return sol


if __name__ == '__main__':
    space_step = 1.e-3
    final_time = .14
    solution = run(space_step, final_time, generator='numpy')
