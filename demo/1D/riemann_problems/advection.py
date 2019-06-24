

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D1Q2 for the advection equation on the 1D-torus

 d_t(u) + c d_x(u) = 0, t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)

 the solution is
 u(t,x) = u0(x-ct).

 test: True
"""
import sympy as sp
import pylbm
from exact_solvers import AdvectionSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

U, X = sp.symbols('u, X')
C, LA, SIGMA = sp.symbols('c, lambda, sigma', constants=True)


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
    xmin, xmax = 0., 1.         # bounds of the domain
    la = 1.                     # lattice velocity (la = dx/dt)
    velocity = 0.25             # velocity of the advection
    s = 1.9                     # relaxation parameter

    symb_s = 1/(0.5+SIGMA)      # symbolic relaxation parameter

    # initial values
    u_left, u_right = 1., 0.
    # discontinuity position
    if velocity > 0:
        xmid = 0.75*xmin + .25*xmax
    elif velocity < 0:
        xmid = .25*xmin + .75*xmax
    else:
        xmid = .5*xmin + .5*xmax
    # fixed bounds of the graphics
    ymin = min([u_left, u_right])-.2*abs(u_left-u_right)
    ymax = max([u_left, u_right])+.2*abs(u_left-u_right)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [u_left],
        'right state': [u_right],
        'velocity': velocity,
    })

    # dictionary of the simulation
    simu_cfg = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': U,
                'polynomials': [1, X],
                'relaxation_parameters': [0., symb_s],
                'equilibrium': [U, C*U],
            },
        ],
        'init': {U: (riemann_pb, (xmid, u_left, u_right))},
        'boundary_conditions': {
            0: {'method': {
                0: pylbm.bc.Neumann,
            }, },
        },
        'generator': generator,
        'parameters': {
            LA: la,
            C: velocity,
            SIGMA: 1/s-.5
        },
        'show_code': False,
    }

    # build the simulation
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)
    # build the equivalent PDE
    eq_pde = pylbm.EquivalentEquation(sol.scheme)
    print(eq_pde)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]
        axe.axis(xmin, xmax, ymin, ymax)
        axe.set_label(r'$x$', r'$u$')

        x = sol.domain.x
        l1a = axe.CurveScatter(
            x, sol.m[U],
            color='navy', label=r'$D_1Q_2$',
        )
        l1e = axe.CurveLine(
            x, exact_solution.evaluate(x, sol.t)[0],
            width=1,
            color='black',
            label='exact',
        )
        axe.legend(loc='upper right',
                   shadow=False,
                   frameon=False,
                   )

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:  # time loop
                sol.one_time_step()  # increment the solution of one time step
                l1a.update(sol.m[U])
                l1e.update(exact_solution.evaluate(x, sol.t)[0])
                axe.title = r'advection at $t = {0:f}$'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./256
    final_time = 1.
    solution = run(space_step, final_time, generator="numpy")
