"""
 Solver D3Q6 for the advection equation on the 3D-torus

 d_t(u) + cx d_x(u) + cy d_y(u) + c_z d_z(u) = 0, t > 0, 0 < x,y,z < 1,
 u(t=0,x,y,z) = u0(x,y,z),
 u(t,x=0,y,z) = u(t,x=1,y,z) 0 < y,z < 1,
 u(t,x,y=0,z) = u(t,x,y=1,z) 0 < x,z < 1,
 u(t,x,y,z=0) = u(t,x,y,z=1) 0 < x,y < 1,

 the solution is
 u(t,x,y,z) = u0(x-cx*t,y-cy*t,z-cz*t)

 test: True
"""
import numpy as np
import sympy as sp
import pylbm

u, X, Y, Z, LA = sp.symbols("u, X, Y, Z, lambda")


def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.domain.mpi_topo, "advection", "./advection", im)
    h5.set_grid(x, y, z)
    h5.add_scalar("u", sol.m[u])
    h5.save()


def run(dx, Tf, generator="cython", sorder=None, with_plot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # advective velocity
    ux, uy, uz = 0.5, 0.2, 0.1
    # domain of the computation
    xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    def u0(x, y, z):
        xm, ym, zm = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        return 0.5 * np.ones((x.size, y.size, z.size)) + 0.5 * (
            ((x - xm) ** 2 + (y - ym) ** 2 + (z - zm) ** 2) < 0.25**2
        )

    s = 1.0
    la = 1.0

    d = {
        "box": {"x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax], "label": -1},
        "space_step": dx,
        "scheme_velocity": la,
        "schemes": [
            {
                "velocities": list(range(1, 7)),
                "conserved_moments": [u],
                "polynomials": [
                    1,
                    LA * X,
                    LA * Y,
                    LA * Z,
                    X**2 - Y**2,
                    X**2 - Z**2,
                ],
                "equilibrium": [u, ux * u, uy * u, uz * u, 0.0, 0.0],
                "relaxation_parameters": [0.0, s, s, s, s, s],
            },
        ],
        "init": {u: u0},
        "parameters": {LA: la},
        "generator": generator,
    }

    sol = pylbm.Simulation(d, sorder=sorder)

    im = 0
    while sol.t < Tf:
        sol.one_time_step()

        if with_plot:
            im += 1
            save(sol, im)

    return sol


if __name__ == "__main__":
    dx = 1.0 / 128
    Tf = 1.0
    run(dx, Tf)
