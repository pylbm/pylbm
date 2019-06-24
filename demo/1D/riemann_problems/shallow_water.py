

"""
 Solver D1Q2Q2 for the shallow water system on [0, 1]

 d_t(h) + d_x(q)    = 0, t > 0, 0 < x < 1,
 d_t(q) + d_x(q^2/h+gh^2/2) = 0, t > 0, 0 < x < 1,
 h(t=0,x) = h0(x), q(t=0,x) = q0(x),
 d_t(h)(t,x=0) = d_t(h)(t,x=1) = 0
 d_t(q)(t,x=0) = d_t(q)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to visualize the simulation of elementary waves

 test: True
"""
import sympy as sp
import pylbm
from exact_solvers import ShallowWaterSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

H, Q, X = sp.symbols('h, q, X')
LA, G = sp.symbols('lambda, g', constants=True)
SIGMA_H, SIGMA_Q = sp.symbols('sigma_1, sigma_2', constants=True)


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
    xmin, xmax = -1, 1.        # bounds of the domain
    gnum = 1                   # numerical value of g
    la = 2.                    # velocity of the scheme
    s_h, s_u = 1.5, 1.5        # relaxation parameters

    symb_s_h = 1/(.5+SIGMA_H)  # symbolic relaxation parameters
    symb_s_q = 1/(.5+SIGMA_Q)  # symbolic relaxation parameters

    # initial values
    h_left, h_right, q_left, q_right = 2, 1, .1, 0.
    # fixed bounds of the graphics
    ymina, ymaxa = 0.9, 2.1
    yminb, ymaxb = -.1, .7
    # discontinuity position
    xmid = .5*(xmin+xmax)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [h_left, q_left],
        'right state': [h_right, q_right],
        'g': gnum,
    })

    exact_solution.diagram()

    simu_cfg = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': H,
                'polynomials': [1, X],
                'relaxation_parameters': [0, symb_s_h],
                'equilibrium': [H, Q],
            },
            {
                'velocities': [1, 2],
                'conserved_moments': Q,
                'polynomials': [1, X],
                'relaxation_parameters': [0, symb_s_q],
                'equilibrium': [Q, Q**2/H+.5*G*H**2],
            },
        ],
        'init': {H: (riemann_pb, (xmid, h_left, h_right)),
                 Q: (riemann_pb, (xmid, q_left, q_right))},
        'boundary_conditions': {
            0: {'method': {
                0: pylbm.bc.Neumann,
                1: pylbm.bc.Neumann,
            }, },
        },
        'generator': generator,
        'parameters': {
            LA: la,
            G: gnum,
            SIGMA_H: 1/s_h-.5,
            SIGMA_Q: 1/s_u-.5,
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
        axe1.set_label(None, "height")
        axe1.xaxis_set_visible(False)
        axe2 = fig[1]
        axe2.axis(xmin, xmax, yminb, ymaxb)
        axe2.set_label(r"$x$", "velocity")

        x = sol.domain.x
        l1a = axe1.CurveScatter(
            x, sol.m[H],
            color='navy', label=r'$D_1Q_2$',
        )
        sole = exact_solution.evaluate(x, sol.t)
        l1e = axe1.CurveLine(
            x, sole[0],
            width=1, color='black', label='exact',
        )
        l2a = axe2.CurveScatter(
            x, sol.m[Q]/sol.m[H],
            color='orange', label=r'$D_1Q_2$',
        )
        l2e = axe2.CurveLine(
            x, sole[1]/sole[0],
            width=1, color='black', label='exact',
        )
        axe1.legend(loc='upper right')
        axe2.legend(loc='upper left')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                sol.one_time_step()
                l1a.update(sol.m[H])
                l2a.update(sol.m[Q]/sol.m[H])
                sole = exact_solution.evaluate(x, sol.t)
                l1e.update(sole[0])
                l2e.update(sole[1]/sole[0])
                axe1.title = r'shallow water at $t = {0:f}$'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./256
    final_time = .5
    solution = run(space_step, final_time, generator="numpy")
