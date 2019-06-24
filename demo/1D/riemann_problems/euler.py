

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D1Q(2,2,2) for the Euler system on [0, 1]

 d_t(rho)   + d_x(rho u)     = 0, t > 0, 0 < x < 1,
 d_t(rho u) + d_x(rho u^2+p) = 0, t > 0, 0 < x < 1,
 d_t(E)   + d_x((E+p) u)     = 0, t > 0, 0 < x < 1,

 where E and p are linked by E = 1/2 rho u^2 + p/(gamma-1)

 then p = (gamma-1)(E - rho u^2/2)
 rho u^2 + p = (gamma-1)E + rho u^2 (3-gamma)/2
 E + p = gamma E - (gamma-1)/2 rho u^2

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
import pylbm
from exact_solvers import EulerSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

RHO, Q, E, X = sp.symbols('rho, q, E, X')
LA = sp.symbols('lambda', constants=True)
GAMMA = sp.Symbol('gamma', constants=True)
SIGMA_RHO, SIGMA_U, SIGMA_P = sp.symbols(
    'sigma_1, sigma_2, sigma_3', constants=True
)


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

    generator: string
        pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution

    Returns
    -------

    sol
        <class 'pylbm.simulation.Simulation'>

    """
    # parameters
    gamma = 1.4                      # ratio of specific heats
    xmin, xmax = 0., 1.              # bounds of the domain
    la = 3.                          # velocity of the scheme
    s_rho, s_u, s_p = 1.9, 1.5, 1.4  # relaxation parameters

    symb_s_rho = 1/(.5+SIGMA_RHO)    # symbolic relaxation parameter
    symb_s_u = 1/(.5+SIGMA_U)        # symbolic relaxation parameter
    symb_s_p = 1/(.5+SIGMA_P)        # symbolic relaxation parameter

    # initial values
    rho_left, p_left, u_left = 1, 1, 0         # left state
    rho_right, p_right, u_right = 1/8, 0.1, 0  # right state
    q_left = rho_left*u_left
    q_right = rho_right*u_right
    rhoe_left = rho_left*u_left**2 + p_left/(gamma-1.)
    rhoe_right = rho_right*u_right**2 + p_right/(gamma-1.)

    # discontinuity position
    xmid = .5*(xmin+xmax)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [rho_left, u_left, p_left],
        'right state': [rho_right, u_right, p_right],
        'gamma': gamma,
    })

    exact_solution.diagram()

    simu_cfg = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': space_step,
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
        'init': {RHO: (riemann_pb, (xmid, rho_left, rho_right)),
                 Q: (riemann_pb, (xmid, q_left, q_right)),
                 E: (riemann_pb, (xmid, rhoe_left, rhoe_right))},
        'boundary_conditions': {
            0: {
                'method': {
                    0: pylbm.bc.Neumann,
                    1: pylbm.bc.Neumann,
                    2: pylbm.bc.Neumann,
                },
            },
        },
        'parameters': {
            LA: la,
            SIGMA_RHO: 1/s_rho-.5,
            SIGMA_U: 1/s_u-.5,
            SIGMA_P: 1/s_p-.5,
            GAMMA: gamma,
        },
        'generator': generator,
    }

    # build the simulation
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)
    # build the equivalent PDE
    eq_pde = pylbm.EquivalentEquation(sol.scheme)
    print(eq_pde)

    while sol.t < final_time:
        sol.one_time_step()

    if with_plot:
        x = sol.domain.x
        rho_n = sol.m[RHO]
        q_n = sol.m[Q]
        rhoe_n = sol.m[E]
        u_n = q_n/rho_n
        p_n = (gamma-1.)*(rhoe_n - .5*rho_n*u_n**2)
        e_n = rhoe_n/rho_n - .5*u_n**2

        sole = exact_solution.evaluate(x, sol.t)

        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(2, 3)

        fig[0, 0].CurveScatter(x, rho_n, color='navy')
        fig[0, 0].CurveLine(x, sole[0], color='orange')
        fig[0, 0].title = 'mass'
        fig[0, 1].CurveScatter(x, u_n, color='navy')
        fig[0, 1].CurveLine(x, sole[1], color='orange')
        fig[0, 1].title = 'velocity'
        fig[0, 2].CurveScatter(x, p_n, color='navy')
        fig[0, 2].CurveLine(x, sole[2], color='orange')
        fig[0, 2].title = 'pressure'
        fig[1, 0].CurveScatter(x, rhoe_n, color='navy')
        rhoe_e = .5*sole[0]*sole[1]**2 + sole[2]/(gamma-1.)
        fig[1, 0].CurveLine(x, rhoe_e, color='orange')
        fig[1, 0].title = 'energy'
        fig[1, 1].CurveScatter(x, q_n, color='navy')
        fig[1, 1].CurveLine(x, sole[0]*sole[1], color='orange')
        fig[1, 1].title = 'momentum'
        fig[1, 2].CurveScatter(x, e_n, color='navy')
        fig[1, 2].CurveLine(x, sole[2]/sole[0]/(gamma-1), color='orange')
        fig[1, 2].title = 'internal energy'

        fig.show()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./1024
    final_time = .14
    solution = run(space_step, final_time, generator="numpy")
