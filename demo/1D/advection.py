

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solvers D1Q2 and D1Q3 for the advection equation on the 1D-torus

 d_t(u) + c d_x(u) = 0, t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)
"""

import sympy as sp
import numpy as np
import pylbm

# pylint: disable=redefined-outer-name

U, X = sp.symbols('u, X')
C, LA = sp.symbols('c, lambda', constants=True)
SIGMA_0, SIGMA_1, SIGMA_2 = sp.symbols(
    'sigma_0, sigma_1, sigma_2', constants=True
)


def u_init(x, xmin, xmax, reg):
    """
    initial condition
    """
    middle, width = .75*xmin+.25*xmax, 0.125*(xmax-xmin)
    x_left, x_right = middle-width, middle+width
    output = np.zeros(x.shape)

    ind_l = np.where(np.logical_and(x > x_left, x <= middle))
    ind_r = np.where(np.logical_and(x < x_right, x > middle))
    x_sl = (x[ind_l] - x_left - 0.5*width) / (0.5*width)
    x_sl_k = np.copy(x_sl)
    x_sl *= x_sl
    x_sr = (x[ind_r] - middle - 0.5*width) / (0.5*width)
    x_sr_k = np.copy(x_sr)
    x_sr *= x_sr

    def binomial(n, k):
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    cte = 0.
    for k in range(reg+1):
        coeff = (-1)**k * binomial(reg, k) / (2*k+1)
        output[ind_l] += coeff * x_sl_k
        output[ind_r] -= coeff * x_sr_k
        cte += coeff
        x_sl_k *= x_sl
        x_sr_k *= x_sr
    output[ind_l] += cte
    output[ind_r] += cte
    output /= 2*cte
    return output


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
    reg = 0                     # regularity of the initial condition
    xmin, xmax = 0., 1.         # bounds of the domain
    la = 1.                     # lattice velocity (la = dx/dt)
    velocity = 0.25             # velocity of the advection
    s_0 = 2.                    # relaxation parameter for the D1Q2
    s_1, s_2 = 1.4, 1.0         # relaxation parameter for the D1Q3

    symb_s0 = 1/(0.5+SIGMA_0)   # symbolic relaxation parameter
    symb_s1 = 1/(0.5+SIGMA_1)   # symbolic relaxation parameter
    symb_s2 = 1/(0.5+SIGMA_2)   # symbolic relaxation parameter

    # fixed bounds of the graphics
    ymin, ymax = -.1, 1.1

    # dictionary of the simulation for the D1Q2
    simu_cfg_d1q2 = {
        'box': {'x': [xmin, xmax], 'label': -1},
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': [1, 2],
                'conserved_moments': U,
                'polynomials': [1, X],
                'relaxation_parameters': [0., symb_s0],
                'equilibrium': [U, C*U],
            },
        ],
        'init': {U: (u_init, (xmin, xmax, reg))},
        'generator': generator,
        'parameters': {
            LA: la,
            C: velocity,
            SIGMA_0: 1/s_0-.5
        },
        'show_code': False,
    }

    # dictionary of the simulation for the D1Q3
    simu_cfg_d1q3 = {
        'box': {'x': [xmin, xmax], 'label': -1},
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': [0, 1, 2],
                'conserved_moments': U,
                'polynomials': [1, X, X**2/2],
                'relaxation_parameters': [0., symb_s1, symb_s2],
                'equilibrium': [U, C*U, C**2*U/2],
            },
        ],
        'init': {U: (u_init, (xmin, xmax, reg))},
        'generator': generator,
        'parameters': {
            LA: la,
            C: velocity,
            SIGMA_1: 1/s_1-.5,
            SIGMA_2: 1/s_2-.5
        },
        'relative_velocity': [C],
        'show_code': False,
    }

    # build the simulations
    sol_d1q2 = pylbm.Simulation(simu_cfg_d1q2, sorder=sorder)
    sol_d1q3 = pylbm.Simulation(simu_cfg_d1q3, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]
        axe.axis(xmin, xmax, ymin, ymax)
        axe.set_label(r'$x$', r'$u$')

        x_d1q2 = sol_d1q2.domain.x
        l1a = axe.CurveScatter(
            x_d1q2, sol_d1q2.m[U],
            color='navy', label=r'$D_1Q_2$'
        )
        x_d1q3 = sol_d1q3.domain.x
        l1b = axe.CurveScatter(
            x_d1q3, sol_d1q3.m[U],
            color='orange', label=r'$D_1Q_3$'
        )
        axe.legend(loc='upper right',
                   shadow=False,
                   frameon=False,
                   )

        def update(iframe):  # pylint: disable=unused-argument
            # increment the solution of one time step
            if sol_d1q2.t < final_time:
                sol_d1q2.one_time_step()
                l1a.update(sol_d1q2.m[U])
            if sol_d1q3.t < final_time:
                sol_d1q3.one_time_step()
                l1b.update(sol_d1q3.m[U])
            axe.title = r'advection at $t = {0:f}$'.format(sol_d1q2.t)

        fig.animate(update)
        fig.show()
    else:
        while sol_d1q2.t < final_time:
            sol_d1q2.one_time_step()

    return sol_d1q2

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./256
    final_time = 2.
    solution = run(space_step, final_time)
