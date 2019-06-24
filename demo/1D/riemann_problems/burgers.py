

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D1Q2 and D1Q3 for the Burger's equation on [-1, 1]

 d_t(u) + d_x(u^2/2) = 0, t > 0, 0 < x < 1,
 u(t=0,x) = u0(x),
 d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0

 the initial condition is a Riemann problem,
 that is a picewise constant function

 u0(x) = uL if x<0, uR if x>0.

 The solution is a shock wave if uL>uR and a linear rarefaction wave if uL<uR

 test: True
"""
import sympy as sp
import pylbm
from exact_solvers import BurgersSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

U, X = sp.symbols('u, X')
LA, SIGMA = sp.symbols('lambda, sigma', constants=True)


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
    xmin, xmax = -1., 1.    # bounds of the domain
    la = 1.                 # scheme velocity (la = dx/dt)
    s = 1.8                 # relaxation parameter

    symb_s = 1/(0.5+SIGMA)  # symbolic relaxation parameter

    # initial values
    u_left, u_right = .0, .3
    # discontinuity position
    xmid = .5*(xmin+xmax)
    # fixed bounds of the graphics
    ymin = min([u_left, u_right])-.1*abs(u_left-u_right)
    ymax = max([u_left, u_right])+.1*abs(u_left-u_right)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [u_left],
        'right state': [u_right],
    })

    # dictionary for the D1Q2
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
                'equilibrium': [U, U**2/2],
            },
        ],
        'init': {U: (riemann_pb, (xmid, u_left, u_right))},
        'boundary_conditions': {
            0: {'method': {
                0: pylbm.bc.Neumann
            }, },
        },
        'generator': generator,
        'parameters': {
            LA: la,
            SIGMA: 1/s-.5
        },
        'show_code': False,
    }

    # build the simulation with D1Q2
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

        x = sol.domain.x
        l1a = axe.CurveScatter(
            x, sol.m[U],
            color='navy', label=r'$D_1Q_2$',
        )
        l1e = axe.CurveLine(
            x, exact_solution.evaluate(x, sol.t)[0],
            width=1, color='black',
            label='exact',
        )
        axe.legend(loc='best',
                   shadow=False,
                   frameon=False,
                   )

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:  # time loop
                sol.one_time_step()  # increment the solution
                l1a.update(sol.m[U])
                l1e.update(exact_solution.evaluate(x, sol.t)[0])
                axe.title = r'Burgers at $t = {0:f}$'.format(sol.t)

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
    run(space_step, final_time, generator="cython")
