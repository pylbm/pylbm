"""
test: True
"""
import pylbm
import sympy as sp

X, Y, Z, LA = sp.symbols("X, Y, Z, lambda")
mass, qx, qy, qz = sp.symbols("mass, qx, qy, qz")


def bc_up(f, m, x, y, z):
    m[qx] = 0.01


def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.domain.mpi_topo, "karman", "./karman", im)
    h5.set_grid(x, y, z)
    h5.add_scalar("mass", sol.m[mass])
    qx_n, qy_n, qz_n = sol.m[qx], sol.m[qy], sol.m[qz]
    h5.add_vector("velocity", [qx_n, qy_n, qz_n])
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
    la = 1.0
    Re = 2000
    nu = 5.0 / Re

    s1 = 1.6
    s2 = 1.2
    s4 = 1.6
    s9 = 1.0 / (3 * nu + 0.5)
    s11 = s9
    s14 = 1.2

    r = X**2 + Y**2 + Z**2

    dico = {
        "box": {
            "x": [0.0, 2.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
            "label": [1, 2, -1, -1, -1, -1],
        },
        "elements": [pylbm.Sphere((0.3, 0.5, 0.5), 0.125, 0)],
        "space_step": dx,
        "scheme_velocity": la,
        "schemes": [
            {
                "velocities": list(range(7)) + list(range(19, 27)),
                "conserved_moments": [mass, qx, qy, qz],
                "polynomials": [
                    1,
                    r - 2,
                    0.5 * (15 * r**2 - 55 * r + 32),
                    X,
                    0.5 * (5 * r - 13) * X,
                    Y,
                    0.5 * (5 * r - 13) * Y,
                    Z,
                    0.5 * (5 * r - 13) * Z,
                    3 * X**2 - r,
                    Y**2 - Z**2,
                    X * Y,
                    Y * Z,
                    Z * X,
                    X * Y * Z,
                ],
                "relaxation_parameters": [
                    0,
                    s1,
                    s2,
                    0,
                    s4,
                    0,
                    s4,
                    0,
                    s4,
                    s9,
                    s9,
                    s11,
                    s11,
                    s11,
                    s14,
                ],
                "equilibrium": [
                    mass,
                    -mass + qx**2 + qy**2 + qz**2,
                    -mass,
                    qx,
                    -7.0 / 3 * qx,
                    qy,
                    -7.0 / 3 * qy,
                    qz,
                    -7.0 / 3 * qz,
                    1.0 / 3 * (2 * qx**2 - (qy**2 + qz**2)),
                    qy**2 - qz**2,
                    qx * qy,
                    qy * qz,
                    qz * qx,
                    0,
                ],
            }
        ],
        "init": {mass: 1.0, qx: 0.0, qy: 0.0, qz: 0.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.BouzidiBounceBack}},
            1: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": bc_up},
            2: {"method": {0: pylbm.bc.NeumannX}},
        },
        "parameters": {LA: la},
        "generator": generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    im = 0
    compt = 0
    while sol.t < Tf:
        sol.one_time_step()
        compt += 1
        if compt == 128 and with_plot:
            im += 1
            save(sol, im)
            compt = 0

    return sol


if __name__ == "__main__":
    dx = 1.0 / 128
    Tf = 200.0
    run(dx, Tf)
