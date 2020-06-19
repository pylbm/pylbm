

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solvers D1Q2 and D1Q3 for the heat equation on (0,1)

 d_t(u) = mu d_xx(u), t > 0, 0 < x < 1,
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1) = 0
"""

import sympy as sp
import numpy as np
import pylbm

# pylint: disable=redefined-outer-name

U, X, DX = sp.symbols('u, X, Delta_x')
MU, LA = sp.symbols('mu, lambda', constants=True)
SIGMA_0, SIGMA_1, SIGMA_2 = sp.symbols(
    'sigma_0, sigma_1, sigma_2', constants=True
)


def solution(x, t, xmin, xmax):
    """
    solution as a eigenvector
    """
    return np.sin(
        np.pi*(x-xmin)/(xmax-xmin)
    )*np.exp(
        -(np.pi/(xmax-xmin))**2*thermal_diffusivity*t
    )


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
    xmin, xmax = 0., 1.         # bounds of the domain
    mu = .1                     # lattice diffusivity (mu = dx^2/dt)
    sigma_0 = thermal_diffusivity / mu  # Henon parameter
    s_0 = 1./(.5+sigma_0)       # relaxation parameter for the D1Q2
    s_1, s_2 = s_0, 1.0         # relaxation parameter for the D1Q3

    symb_s0 = 1/(0.5+SIGMA_0)   # symbolic relaxation parameter
    symb_s1 = 1/(0.5+SIGMA_1)   # symbolic relaxation parameter
    symb_s2 = 1/(0.5+SIGMA_2)   # symbolic relaxation parameter

    # fixed bounds of the graphics
    ymin, ymax = -.1, 1.1

    # dictionary of the simulation for the D1Q2
    simu_cfg_d1q2 = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': DX,
        'lattice_diffusivity': MU,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': U,
                'polynomials': [1, X],
                'relaxation_parameters': [0., symb_s0],
                'equilibrium': [U, 0],
            },
        ],
        # 'init': {U: (u_init, (xmin, xmax, reg))},
        'init': {U: (solution, (0, xmin, xmax))},
        'generator': generator,
        'parameters': {
            DX: space_step,
            MU: mu,
            SIGMA_0: 1/s_0-.5
        },
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiAntiBounceBack}},
        },
        'show_code': False,
    }

    # dictionary of the simulation for the D1Q3
    simu_cfg_d1q3 = {
        'box': {'x': [xmin, xmax], 'label': 0},
        'space_step': space_step,
        'lattice_diffusivity': MU,
        'schemes': [
            {
                'velocities': [0, 1, 2],
                'conserved_moments': U,
                'polynomials': [1, X, X**2/2],
                'relaxation_parameters': [0., symb_s1, symb_s2],
                'equilibrium': [U, 0, U/2],
            },
        ],
        # 'init': {U: (u_init, (xmin, xmax, reg))},
        'init': {U: (solution, (0, xmin, xmax))},
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiAntiBounceBack}},
        },
        'generator': generator,
        'parameters': {
            MU: mu,
            SIGMA_1: 1/s_1-.5,
            SIGMA_2: 1/s_2-.5
        },
        'show_code': False,
    }

    # build the simulations
    sol_d1q2 = pylbm.Simulation(simu_cfg_d1q2, sorder=sorder)
    sol_d1q3 = pylbm.Simulation(simu_cfg_d1q3, sorder=sorder)

    if with_plot:
        x_d1q2 = sol_d1q2.domain.x
        x_d1q3 = sol_d1q3.domain.x
        x_exact = np.linspace(xmin, xmax, 1025)
        u_exact = solution(x_exact, sol_d1q2.t, xmin, xmax)
        u_exact_d1q2 = solution(x_d1q2, sol_d1q2.t, xmin, xmax)
        u_exact_d1q3 = solution(x_d1q2, sol_d1q2.t, xmin, xmax)

        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(2, 2, figsize=(12, 12))
        fig.title(r'heat at $t = {0:f}$'.format(sol_d1q2.t))

        ax2 = fig[0, 0]
        ax2.axis(xmin, xmax, ymin, ymax)
        ax2.set_label(r'$x$', r'$u$')
        ax2.set_title('Solution for $D_1Q_2$')
        l2a = ax2.CurveScatter(
            x_d1q2, sol_d1q2.m[U],
            color='navy', size=25,
            label=r'$D_1Q_2$'
        )
        l2e = ax2.CurveLine(
            x_exact, u_exact,
            color='black', alpha=0.5, width=2,
            label='exact'
        )
        ax2.legend(
            loc='upper right',
            shadow=False,
            frameon=False,
        )

        ax3 = fig[0, 1]
        ax3.axis(xmin, xmax, ymin, ymax)
        ax3.set_label(r'$x$', r'$u$')
        ax3.set_title('Solution for $D_1Q_2$')
        l3a = ax3.CurveScatter(
            x_d1q3, sol_d1q3.m[U],
            color='orange', size=25,
            label=r'$D_1Q_3$'
        )
        l3e = ax3.CurveLine(
            x_exact, u_exact,
            color='black', alpha=0.5, width=2,
            label='exact'
        )
        ax3.legend(
            loc='upper right',
            shadow=False,
            frameon=False,
        )

        ax2e = fig[1, 0]
        ax2e.axis(xmin, xmax, -1.e-3, 1.e-3)
        ax2e.set_label(r'$x$', r'erreur')
        ax2e.set_title('Error for $D_1Q_2$')
        l2er = ax2e.CurveScatter(
            x_d1q2, sol_d1q2.m[U] - u_exact_d1q2,
            color='navy', size=25,
            label=r'$D_1Q_2$'
        )
        ax3e = fig[1, 1]
        ax3e.axis(xmin, xmax, -1.e-3, 1.e-3)
        ax3e.set_label(r'$x$', r'erreur')
        ax3e.set_title('Error for $D_1Q_3$')
        l3er = ax3e.CurveScatter(
            x_d1q3, sol_d1q3.m[U] - u_exact_d1q3,
            color='orange', size=25,
            label=r'$D_1Q_3$'
        )

        def update(iframe):  # pylint: disable=unused-argument
            # increment the solution of one time step
            loc_tf = min(final_time, final_time*iframe/10)
            while sol_d1q2.t < loc_tf:
                sol_d1q2.one_time_step()
            u_exact_d1q2[:] = solution(x_d1q2, sol_d1q2.t, xmin, xmax)
            l2a.update(sol_d1q2.m[U])
            l2er.update(sol_d1q2.m[U] - u_exact_d1q2)
            while sol_d1q3.t < loc_tf:
                sol_d1q3.one_time_step()
            u_exact_d1q3[:] = solution(x_d1q3, sol_d1q3.t, xmin, xmax)
            l3a.update(sol_d1q3.m[U])
            l3er.update(sol_d1q3.m[U] - u_exact_d1q3)
            u_exact[:] = solution(x_exact, sol_d1q2.t, xmin, xmax)
            l2e.update(u_exact)
            l3e.update(u_exact)
            fig.title(r'heat at $t = {0:f}$'.format(sol_d1q3.t))

        fig.animate(update)
        fig.show()
    else:
        while sol_d1q2.t < final_time:
            sol_d1q2.one_time_step()

    return sol_d1q2


if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./64
    thermal_diffusivity = 0.1
    final_time = 1.
    solution = run(space_step, final_time)
