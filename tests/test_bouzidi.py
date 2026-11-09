"""
BFL (Bouzidi-Firdaouss-Lallemand) conditions on a wall at a distance delta of the last fluid node.
The steady solutions are linear, which these schemes reproduce exactly.
"""

import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y, LA = sp.symbols("X, Y, lambda")
rho, qx, qy, u = sp.symbols("rho, qx, qy, u")

DX = 1.0 / 8
HEIGHT = 1.0
DELTAS = [0.25, 0.5, 0.75, 0.9]


def wall_position(delta):
    """y of a wall at delta from the last fluid node, near y = HEIGHT."""
    return HEIGHT - 1.5 * DX + delta * DX


def top_wall(yw):
    return pylbm.Parallelogram((-1.0, yw), (3.0, 0.0), (0.0, 1.0), label=1)


def run_to_steady_state(sol, moment):
    previous = None
    for _ in range(200):
        for _ in range(200):
            sol.one_time_step()
        current = np.array(sol.m[moment])
        if previous is not None and np.max(np.abs(current - previous)) < 1e-14:
            return current
        previous = current
    raise RuntimeError("no steady state")


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("x_boundary", ["periodic", "profile"])
def test_bouzidi_bounce_back_couette(generator, delta, x_boundary):
    """
    Couette flow between a fixed wall (y = 0) and a wall moving at uw, at a
    distance delta of the last fluid node. The x direction is either
    periodic, or closed by an inlet and an outlet that impose the exact
    profile. The periodic case checks the diagonal links at the corners,
    which leave the domain through the wall and not through the periodic
    face.
    """
    uw = 0.05
    yw = wall_position(delta)
    nu = 0.1
    s_plus = 1.0 / (0.5 + 3.0 * nu / DX)
    s_minus = 1.0 / (0.5 + (3.0 / 16.0) / (1.0 / s_plus - 0.5))

    def velocity(value):
        def wall(f, m, x, y):
            m[rho] = 1.0
            m[qx] = value(y)
            m[qy] = 0.0

        return wall

    x_label = -1 if x_boundary == "periodic" else 2
    dico = {
        "box": {
            "x": [0.0, 0.5],
            "y": [0.0, HEIGHT],
            "label": [x_label, x_label, 0, 1],
        },
        "elements": [top_wall(yw)],
        "space_step": DX,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(9)),
                "polynomials": [
                    1,
                    LA * X,
                    LA * Y,
                    3 * (X**2 + Y**2) - 4,
                    0.5 * (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8),
                    3 * X * (X**2 + Y**2) - 5 * X,
                    3 * Y * (X**2 + Y**2) - 5 * Y,
                    X**2 - Y**2,
                    X * Y,
                ],
                "relaxation_parameters": [0, 0, 0]
                + [s_plus] * 2
                + [s_minus] * 2
                + [s_plus] * 2,
                "equilibrium": [rho, qx, qy, -2 * rho, rho, -qx / LA, -qy / LA, 0, 0],
                "conserved_moments": [rho, qx, qy],
            }
        ],
        "parameters": {LA: 1.0},
        "init": {rho: 1.0, qx: 0.0, qy: 0.0},
        "boundary_conditions": {
            0: {
                "method": {0: pylbm.bc.BFLBounceBack},
                "value": velocity(lambda y: 0.0),
            },
            1: {
                "method": {0: pylbm.bc.BFLBounceBack},
                "value": velocity(lambda y: uw),
            },
        },
        "generator": generator,
    }
    if x_boundary == "profile":
        dico["boundary_conditions"][2] = {
            "method": {0: pylbm.bc.BFLBounceBack},
            "value": velocity(lambda y: uw * y / yw),
        }
    sol = pylbm.Simulation(dico)
    ux = run_to_steady_state(sol, qx)

    fluid = sol.domain.y < yw
    exact = uw * sol.domain.y[fluid] / yw
    np.testing.assert_allclose(
        ux[:, fluid], np.broadcast_to(exact, ux[:, fluid].shape), atol=1e-12
    )


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("delta", DELTAS)
def test_bouzidi_anti_bounce_back_diffusion(generator, delta):
    """
    Steady diffusion between u = 0 at y = 0 and u = 1 on a wall at a distance
    delta of the last fluid node, with a D2Q5 TRT scheme.
    """
    yw = wall_position(delta)

    def value(c):
        def wall(f, m, x, y):
            m[u] = c

        return wall

    dico = {
        "box": {"x": [0.0, 0.5], "y": [0.0, HEIGHT], "label": [-1, -1, 0, 1]},
        "elements": [top_wall(yw)],
        "space_step": DX,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0.0, 0.8, 0.8, 1.3, 1.3],
                "equilibrium": [u, 0, 0, 0, 0.5 * u],
            }
        ],
        "parameters": {LA: 1.0},
        "init": {u: 0.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.BFLAntiBounceBack}, "value": value(0.0)},
            1: {"method": {0: pylbm.bc.BFLAntiBounceBack}, "value": value(1.0)},
        },
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    c = run_to_steady_state(sol, u)

    fluid = sol.domain.y < yw
    exact = sol.domain.y[fluid] / yw
    np.testing.assert_allclose(
        c[:, fluid], np.broadcast_to(exact, c[:, fluid].shape), atol=1e-12
    )


def advection_diffusion_with_corners(method, generator):
    """
    Advection-diffusion with a d2Q9 scheme in a box that holds a square
    obstacle, a disk and a notch in the bottom wall. Near the disk, a link
    reads a population in the solid that another link of the condition
    writes.
    """
    r2 = X**2 + Y**2

    def value(f, m, x, y):
        m[u] = 1.0 + x * y

    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": 0},
        "elements": [
            pylbm.Parallelogram([0.4, 0.4], [0.23, 0], [0, 0.17], label=0),
            pylbm.Parallelogram([0.0, 0.0], [0.3, 0], [0, 0.21], label=0),
            pylbm.Circle([0.7, 0.3], 0.13, label=0),
        ],
        "space_step": 1.0 / 32,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(9)),
                "conserved_moments": u,
                "polynomials": [
                    1,
                    X,
                    Y,
                    3 * r2 - 4,
                    (9 * r2**2 - 21 * r2 + 8) / 2,
                    3 * X * r2 - 5 * X,
                    3 * Y * r2 - 5 * Y,
                    X**2 - Y**2,
                    X * Y,
                ],
                "relaxation_parameters": [0, 1.5, 1.5, 1.2, 1.2, 1.5, 1.5, 1.2, 1.2],
                "equilibrium": [u, 0.05 * u, 0.02 * u, -2 * u, u, 0, 0, 0, 0],
            }
        ],
        "parameters": {LA: 1.0},
        "init": {u: 0.0},
        "boundary_conditions": {0: {"method": {0: method}, "value": value}},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    for _ in range(50):
        sol.one_time_step()
    return np.array(sol.F_halo[:])[:, 1:-1, 1:-1]


@pytest.mark.parametrize("method", [pylbm.bc.BFLBounceBack, pylbm.bc.BFLAntiBounceBack])
def test_bouzidi_corners_numpy_and_cython(method):
    """NumPy and Cython give the same populations near the corners."""
    np.testing.assert_allclose(
        advection_diffusion_with_corners(method, "cython"),
        advection_diffusion_with_corners(method, "numpy"),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("method", [pylbm.bc.BFLBounceBack, pylbm.bc.BFLAntiBounceBack])
def test_bouzidi_corners_loopy(method):
    np.testing.assert_allclose(
        advection_diffusion_with_corners(method, "loopy"),
        advection_diffusion_with_corners(method, "numpy"),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "old_name, new_name",
    [
        ("BouzidiBounceBack", "BFLBounceBack"),
        ("BouzidiAntiBounceBack", "BFLAntiBounceBack"),
    ],
)
def test_former_names_are_deprecated(old_name, new_name):
    """The former names still give the classes, with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match=new_name):
        method = getattr(pylbm.bc, old_name)
    assert method is getattr(pylbm.bc, new_name)
