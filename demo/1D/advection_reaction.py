

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D1Q2 for the advection reaction equation on the 1D-torus

 d_t(u) + c d_x(u) = mu u(1-u), t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)

 test: True
"""
import sympy as sp
import numpy as np
import pylbm

# pylint: disable=redefined-outer-name

X, U = sp.symbols('X, u')
C, MU, LA = sp.symbols('c, mu, lambda', constants=True)


def u_init(x, xmin, xmax):
    """
    initial condition
    """
    middle = 0.5*(xmin+xmax)
    width = 0.1*(xmax-xmin)
    x_centered = xmin + x % (xmax-xmin)
    middle = 0.5*(xmin+xmax)
    return 0.25 \
        + .125/width**10 * (x_centered-middle-width)**5 \
        * (middle-x_centered-width)**5 \
        * (abs(x_centered-middle) <= width)


def solution(t, x, xmin, xmax, c, mu):
    """
    exact solution
    """
    dt = np.tanh(0.5*mu*t)
    u_i = u_init(x - c*t, xmin, xmax)
    return (dt+2*u_i-(1-2*u_i)*dt)/(2-2*(1-2*u_i)*dt)


def run(space_step,
        final_time,
        generator="numpy",
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
    xmin, xmax = 0., 1.   # bounds of the domain
    la = 1.               # scheme velocity (la = dx/dt)
    c = 0.25              # velocity of the advection
    mu = 1.               # parameter of the source term
    s = 2.                # relaxation parameter

    # dictionary of the simulation
    simu_cfg = {
        'box': {'x': [xmin, xmax], 'label': -1},
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': U,
                'polynomials': [1, X],
                'relaxation_parameters': [0., s],
                'equilibrium': [U, C*U],
                'source_terms': {U: MU*U*(1-U)},
            },
        ],
        'init': {U: (u_init, (xmin, xmax))},
        'generator': generator,
        'parameters': {LA: la, C: c, MU: mu},
    }

    # build the simulation
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]
        ymin, ymax = -.2, 1.2
        axe.axis(xmin, xmax, ymin, ymax)

        x = sol.domain.x
        l1a = axe.CurveScatter(
            x, sol.m[U],
            color='navy', label='D1Q2'
        )
        l1e = axe.CurveLine(
            x, solution(sol.t, x, xmin, xmax, c, mu),
            label='exact'
        )
        axe.legend()

        def update(iframe):  # pylint: disable=unused-argument
            # increment the solution of one time step
            if sol.t < final_time:
                sol.one_time_step()
                l1a.update(sol.m[U])
                l1e.update(solution(sol.t, x, xmin, xmax, c, mu))
                axe.title = 'solution at t = {0:f}'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = 2.
    solution = run(space_step, final_time)
