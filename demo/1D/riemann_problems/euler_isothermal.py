

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D1Q(2,2) for the isothermal Euler system on [0, 1]

 d_t(rho)   + d_x(rho u)            = 0, t > 0, 0 < x < 1,
 d_t(rho u) + d_x(rho u^2+c_0^2rho) = 0, t > 0, 0 < x < 1,

 where c_0 is the speed of sound

 Initial and boundary conditions are:

 rho(t=0,x) = rho0(x), u(t=0,x) = u0(x)
 d_t(rho)(t,x=0) = d_t(rho)(t,x=1) = 0
 d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to simulate a Riemann problem

 test: True
"""

from numpy import sqrt
import sympy as sp
import pylbm
from exact_solvers import EulerIsothermalSolver as exact_solver
from exact_solvers import riemann_pb

# pylint: disable=redefined-outer-name

RHO, Q, X = sp.symbols('rho, q, X')
LA, CO = sp.symbols('lambda, c_0', constants=True)
SIGMA_RHO, SIGMA_U = sp.symbols('sigma_1, sigma_2', constants=True)


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
    xmin, xmax = -1, 1               # bounds of the domain
    la = 1.                          # velocity of the scheme
    c_0 = la/sqrt(3)                 # velocity of the pressure waves
    s_rho, s_u = 2., 1.85            # relaxation parameters

    symb_s_rho = 1/(.5+SIGMA_RHO)    # symbolic relaxation parameter
    symb_s_u = 1/(.5+SIGMA_U)        # symbolic relaxation parameter

    # initial values
    rhoo = 1
    drho = rhoo/3
    rho_left, u_left = rhoo-drho, 0    # left state
    rho_right, u_right = rhoo+drho, 0  # right state
    q_left = rho_left*u_left
    q_right = rho_right*u_right
    # fixed bounds of the graphics
    ymina, ymaxa = rhoo-2*drho, rhoo+2*drho
    yminb, ymaxb = -.25, .1

    # discontinuity position
    xmid = .5*(xmin+xmax)

    exact_solution = exact_solver({
        'jump abscissa': xmid,
        'left state': [rho_left, u_left],
        'right state': [rho_right, u_right],
        'sound_speed': c_0,
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
                'equilibrium': [Q, Q**2/RHO + CO**2*RHO],
            },
        ],
        'init': {RHO: (riemann_pb, (xmid, rho_left, rho_right)),
                 Q: (riemann_pb, (xmid, q_left, q_right))},
        'boundary_conditions': {
            0: {
                'method': {
                    0: pylbm.bc.Neumann,
                    1: pylbm.bc.Neumann,
                },
            },
        },
        'parameters': {
            LA: la,
            SIGMA_RHO: 1/s_rho-.5,
            SIGMA_U: 1/s_u-.5,
            CO: c_0,
        },
        'generator': generator,
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
        axe1.set_label(None, r'$\rho$')
        axe1.xaxis_set_visible(False)
        axe2 = fig[1]
        axe2.axis(xmin, xmax, yminb, ymaxb)
        axe2.set_label(r"$x$", r"$u$")

        x = sol.domain.x
        l1a = axe1.CurveScatter(
            x, sol.m[RHO],
            color='navy',
            label=r'$D_1Q_2$',
        )
        sole = exact_solution.evaluate(x, sol.t)
        l1e = axe1.CurveLine(
            x, sole[0], width=1,
            label='exact',
        )
        l2a = axe2.CurveScatter(
            x, sol.m[Q]/sol.m[RHO],
            color='orange',
            label=r'$D_1Q_2$',
        )
        l2e = axe2.CurveLine(
            x, sole[1], width=1,
            label='exact',
        )
        axe1.legend(loc='lower right')
        axe2.legend(loc='lower right')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                sol.one_time_step()
                l1a.update(sol.m[RHO])
                l2a.update(sol.m[Q]/sol.m[RHO])
                sole = exact_solution.evaluate(x, sol.t)
                l1e.update(sole[0])
                l2e.update(sole[1])
                axe1.title = r'isothermal Euler at $t = {0:f}$'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = sqrt(3)/2
    solution = run(space_step, final_time, generator="numpy")
