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

U, X, Y = sp.symbols("U, X, Y")
CX, CY, LA = sp.symbols("c_0, c_1, lambda", constants=True)
SIGMA_0, SIGMA_1 = sp.symbols("sigma_0, sigma_1", constants=True)

hmin, hmax = 1, 1.5


def u_init(x, y, xmin, xmax, ymin, ymax):
    """
    initial condition
    """
    center = (0.75 * xmin + 0.25 * xmax, 0.50 * ymin + 0.50 * ymax)
    radius = 0.1
    return hmin + (hmax - hmin) * (
        (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
    )


def f_init(x, y, k, cx, cy, xmin, xmax, ymin, ymax):
    coeff = 0.25
    if k == 0:
        coeff += 0.5 * cx
    elif k == 1:
        coeff += 0.5 * cy
    elif k == 2:
        coeff -= 0.5 * cx
    elif k == 3:
        coeff -= 0.5 * cy
    else:
        coeff = 0
    return coeff * u_init(x, y, xmin, xmax, ymin, ymax)


def run(space_step, final_time, generator="cython", sorder=None, with_plot=True):
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
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0  # bounds of the domain
    c_x, c_y = 0.2, 0.5  # velocity of the advection
    la = 2.0  # scheme velocity
    sigma_q = 1.0e-2  # 1./np.sqrt(12)
    sigma_xy = 0.5  # sigma_q
    symb_sq = 1 / (0.5 + SIGMA_0)
    symb_sxy = 1 / (0.5 + SIGMA_1)
    s = [0.0, symb_sq, symb_sq, symb_sxy]  # relaxation parameters

    simu_cfg = {
        "box": {"x": [xmin, xmax], "y": [ymin, ymax], "label": -1},
        "space_step": space_step,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(1, 5)),
                "conserved_moments": U,
                "polynomials": [1, X, Y, X**2 - Y**2],
                "relaxation_parameters": s,
                "equilibrium": [U, CX * U, CY * U, 0],
            },
        ],
        "inittype": "distributions",
        "init": {
            0: (f_init, (0, c_x / la, c_y / la, xmin, xmax, ymin, ymax)),
            1: (f_init, (1, c_x / la, c_y / la, xmin, xmax, ymin, ymax)),
            2: (f_init, (2, c_x / la, c_y / la, xmin, xmax, ymin, ymax)),
            3: (f_init, (3, c_x / la, c_y / la, xmin, xmax, ymin, ymax)),
        },
        # 'inittype': 'moments',
        # 'init': {
        #     U: (u_init, (xmin, xmax, ymin, ymax))
        # },
        "generator": generator,
        "parameters": {
            LA: la,
            CX: c_x,
            CY: c_y,
            SIGMA_0: sigma_q,
            SIGMA_1: sigma_xy,
        },
        "relative_velocity": [CX, CY],
    }

    # build the simulations
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]

        vmin, vmax, dv = hmin, hmax, 0.1 * (hmax - hmin)
        vmin -= dv
        vmax += dv
        im = axe.image(sol.m[U].transpose(), cmap="jet", clim=[vmin, vmax])
        axe.title = f"Advection at t = {sol.t:.2f}"

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                for _ in range(16):
                    sol.one_time_step()
                im.set_data(sol.m[U].transpose())
                axe.title = f"Advection at t = {sol.t:.2f}"

        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol


if __name__ == "__main__":
    # pylint: disable=invalid-name
    space_step = 1.0 / 256
    final_time = 20.0
    solution = run(space_step, final_time)
