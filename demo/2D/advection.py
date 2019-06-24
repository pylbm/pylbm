

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D2Q4 for the advection equation on the 2D-torus

 dt u + cx dx u + cy dy u = 0, t > 0, 0 < x,y < 1,
 u(t=0,x,y) = u0(x,y),
 u(t,x=0,y) = u(t,x=1,y) 0 < y < 1,
 u(t,x,y=0) = u(t,x,y=1) 0 < x < 1,

 the solution is
 u(t,x,y) = u0(x-cx*t,y-cy*t)
"""

import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name

U, X, Y = sp.symbols('U, X, Y')
CX, CY, LA = sp.symbols('c_0, c_1, lambda', constants=True)
SIGMA_0, SIGMA_1 = sp.symbols(
    'sigma_0, sigma_1', constants=True
)


def u_init(x, y, xmin, xmax, ymin, ymax):
    """
    initial condition
    """
    center = (
        .75*xmin + .25*xmax,
        .50*ymin + .50*ymax
    )
    radius = 0.1
    height = 0.5
    return 1 + height * ((x-center[0])**2+(y-center[1])**2 < radius**2)


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
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.  # bounds of the domain
    c_x, c_y = 0.2, 0.5                      # velocity of the advection
    la = 2.                                  # scheme velocity
    sigma_q = 1.e-2  # 1./np.sqrt(12)
    sigma_xy = 0.5   # sigma_q
    symb_sq = 1/(.5+SIGMA_0)
    symb_sxy = 1/(.5+SIGMA_1)
    s = [0., symb_sq, symb_sq, symb_sxy]     # relaxation parameters

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': -1
        },
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': U,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s,
                'equilibrium': [U, CX*U, CY*U, 0],
            },
        ],
        'init': {
            U: (u_init, (xmin, xmax, ymin, ymax))
        },
        'generator': generator,
        'parameters': {
            LA: la,
            CX: c_x,
            CY: c_y,
            SIGMA_0: sigma_q,
            SIGMA_1: sigma_xy,
        },
        'relative_velocity': [CX, CY],
        }

    # build the simulations
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(dim=3)
        axe = fig[0]

        surf = axe.SurfaceScatter(
            sol.domain.x, sol.domain.y, sol.m[U],
            size=2, sampling=4, color='navy'
        )
        axe.title = 'Advection'
        axe.grid(visible=False)
        axe.set_label(r'$x$', r'$y$', r'$u$')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                for _ in range(16):
                    sol.one_time_step()
                surf.update(sol.m[U])

        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./256
    final_time = 2.
    solution = run(space_step, final_time)
