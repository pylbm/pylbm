

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D2Q4Q4Q4 for the shallow water equation on the 2D-torus

 dt h + dx q_x + dy q_y = 0,
 dt q_x + dx (q_x^2/h + gh^2/2) + dy (q_xq_y/h) = 0,
 dt q_y + dx (q_xq_y/h) + dy (q_y^2/h + gh^2/2) = 0,
"""
import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name

X, Y = sp.symbols('X, Y')
H, QX, QY = sp.symbols('h, qx, qy')
LA, G = sp.symbols('lambda, g', constants=True)
SIGMA_HX, SIGMA_HXY, SIGMA_QX, SIGMA_QXY = sp.symbols(
    'sigma_0, sigma_1, sigma_2, sigma_3', constants=True
)


def h_init(x, y, xmin, xmax, ymin, ymax):
    """
    initial condition
    """
    center = (
        .5*xmin + .5*xmax,
        .5*ymin + .5*ymax
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
    xmin, xmax, ymin, ymax = -1., 1., -1., 1.  # bounds of the domain
    la = 4                                     # velocity of the scheme
    gravity = 1.                               # gravity
    sigma_hx = 1.e-3
    sigma_hxy = 0.5
    sigma_qx = 1.e-1
    sigma_qxy = 0.5
    symb_s_hx = 1/(.5+SIGMA_HX)
    symb_s_hxy = 1/(.5+SIGMA_HXY)
    symb_s_qx = 1/(.5+SIGMA_QX)
    symb_s_qxy = 1/(.5+SIGMA_QXY)

    s_h = [0., symb_s_hx, symb_s_hx, symb_s_hxy]
    s_q = [0., symb_s_qx, symb_s_qx, symb_s_qxy]

    vitesse = list(range(1, 5))
    polynomes = [1, X, Y, X**2-Y**2]

    simu_cfg = {
        'parameters': {
            LA: la,
            G: gravity,
            SIGMA_HX: sigma_hx,
            SIGMA_HXY: sigma_hxy,
            SIGMA_QX: sigma_qx,
            SIGMA_QXY: sigma_qxy,
        },
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': -1
        },
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': vitesse,
                'conserved_moments': H,
                'polynomials': polynomes,
                'relaxation_parameters': s_h,
                'equilibrium': [H, QX, QY, 0.],
            },
            {
                'velocities': vitesse,
                'conserved_moments': QX,
                'polynomials': polynomes,
                'relaxation_parameters': s_q,
                'equilibrium': [QX, QX**2/H + G*H**2/2, QX*QY/H, 0.],
            },
            {
                'velocities': vitesse,
                'conserved_moments': QY,
                'polynomials': polynomes,
                'relaxation_parameters': s_q,
                'equilibrium': [QY, QX*QY/H, QY**2/H + G*H**2/2, 0.],
            },
        ],
        'init': {H: (h_init, (xmin, xmax, ymin, ymax)),
                 QX: 0.,
                 QY: 0.},
        'relative_velocity': [QX/H, QY/H],
        'generator': generator,
        }

    # build the simulations
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig(dim=3)
        axe = fig[0]

        surf = axe.SurfaceScatter(
            sol.domain.x, sol.domain.y, sol.m[H],
            size=2, sampling=4, color='navy'
        )
        axe.title = 'Shallow water'
        axe.grid(visible=False)
        axe.set_label(r'$x$', r'$y$', r'$h$')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                for _ in range(16):
                    sol.one_time_step()
                surf.update(sol.m[H])

        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = 20
    run(space_step, final_time)
