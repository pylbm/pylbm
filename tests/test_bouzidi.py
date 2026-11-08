"""
Bouzidi conditions on a wall at a distance delta of the last fluid node.
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
def test_bouzidi_bounce_back_couette(generator, delta):
    """
    Couette flow between a fixed wall (y = 0) and a wall moving at uw, at a
    distance delta of the last fluid node. The inlet and the outlet impose
    the exact profile, so that every cut link has a consistent value.
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

    dico = {
        "box": {"x": [0.0, 0.5], "y": [0.0, HEIGHT], "label": [2, 2, 0, 1]},
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
                "method": {0: pylbm.bc.BouzidiBounceBack},
                "value": velocity(lambda y: 0.0),
            },
            1: {
                "method": {0: pylbm.bc.BouzidiBounceBack},
                "value": velocity(lambda y: uw),
            },
            2: {
                "method": {0: pylbm.bc.BouzidiBounceBack},
                "value": velocity(lambda y: uw * y / yw),
            },
        },
        "generator": generator,
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
            0: {"method": {0: pylbm.bc.BouzidiAntiBounceBack}, "value": value(0.0)},
            1: {"method": {0: pylbm.bc.BouzidiAntiBounceBack}, "value": value(1.0)},
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
