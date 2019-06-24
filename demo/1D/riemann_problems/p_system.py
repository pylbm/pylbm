

"""
 Solver D1Q2Q2 for the p-system on [0, 1]

 d_t(u1) - d_x(u2)    = 0, t > 0, 0 < x < 1,
 d_t(u2) - d_x(p(u1)) = 0, t > 0, 0 < x < 1,
 u1(t=0,x) = u1_0(x), u2(t=0,x) = u2_0(x),
 d_t(u1)(t,x=0) = d_t(u1)(t,x=1) = 0
 d_t(u2)(t,x=0) = d_t(u2)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to visualize the simulation of elementary waves

 test: True
"""

import sympy as sp
import pylbm
from exact_solvers import PSystemSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

U1, U2, X = sp.symbols('u_1, u_2, X')
LA, SIGMA_1, SIGMA_2 = sp.symbols('lambda, sigma_1, sigma_2', constants=True)
GAMMA = sp.Symbol('gamma', constants=True)


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
    xmin, xmax = 0., 1.  # bounds of the domain
    gamma = 2./3.        # exponent in the p-function
    la = 2.              # velocity of the scheme
    s_1, s_2 = 1.9, 1.9  # relaxation parameters

    symb_s_1 = 1/(.5+SIGMA_1)  # symbolic relaxation parameters
    symb_s_2 = 1/(.5+SIGMA_2)  # symbolic relaxation parameters

    # initial values
    u1_left, u1_right, u2_left, u2_right = 1.50, 0.50, 1.25, 1.50
    # fixed bounds of the graphics
    ymina, ymaxa, yminb, ymaxb = .25, 1.75, 0.75, 1.75
    # discontinuity position
    xmid = .5*(xmin+xmax)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [u1_left, u2_left],
        'right state': [u1_right, u2_right],
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
                'conserved_moments': U1,
                'polynomials': [1, X],
                'relaxation_parameters': [0, symb_s_1],
                'equilibrium': [U1, -U2],
            },
            {
                'velocities': [1, 2],
                'conserved_moments': U2,
                'polynomials': [1, X],
                'relaxation_parameters': [0, symb_s_2],
                'equilibrium': [U2, U1**(-GAMMA)],
            },
        ],
        'init': {U1: (riemann_pb, (xmid, u1_left, u1_right)),
                 U2: (riemann_pb, (xmid, u2_left, u2_right))},
        'boundary_conditions': {
            0: {'method': {
                0: pylbm.bc.Neumann,
                1: pylbm.bc.Neumann,
            }, },
        },
        'generator': generator,
        'parameters': {
            LA: la,
            GAMMA: gamma,
            SIGMA_1: 1/s_1-.5,
            SIGMA_2: 1/s_2-.5,
        },
    }

    # build the simulation
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)
    # build the equivalent PDE
    eq_pde = pylbm.EquivalentEquation(sol.scheme)
    print(eq_pde)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(2, 1)
        axe1 = fig[0]
        axe1.axis(xmin, xmax, ymina, ymaxa)
        axe1.set_label(None, r"$u_1$")
        axe1.xaxis_set_visible(False)
        axe2 = fig[1]
        axe2.axis(xmin, xmax, yminb, ymaxb)
        axe2.set_label(r"$x$", r"$u_2$")

        x = sol.domain.x
        l1a = axe1.CurveScatter(
            x, sol.m[U1],
            color='navy', label=r'$D_1Q_2$',
        )
        sole = exact_solution.evaluate(x, sol.t)
        l1e = axe1.CurveLine(
            x, sole[0],
            width=1, color='black', label='exact',
        )
        l2a = axe2.CurveScatter(
            x, sol.m[U2],
            color='orange', label=r'$D_1Q_2$',
        )
        l2e = axe2.CurveLine(
            x, sole[1],
            width=1, color='black', label='exact',
        )
        axe1.legend(loc='upper right')
        axe2.legend(loc='upper left')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                sol.one_time_step()
                l1a.update(sol.m[U1])
                l2a.update(sol.m[U2])
                sole = exact_solution.evaluate(x, sol.t)
                l1e.update(sole[0])
                l2e.update(sole[1])
                axe1.title = r'p-system at $t = {0:f}$'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./256
    final_time = .25
    solution = run(space_step, final_time, generator="numpy")
