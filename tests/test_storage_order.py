import itertools

import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y, Z, LA = sp.symbols("X, Y, Z, lambda")
u = sp.symbols("u")


def advection_2d(generator, sorder):
    """D2Q5 advection on an anisotropic periodic domain."""
    cx, cy = 0.3, -0.2
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "label": -1},
        "space_step": 0.125,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0.0, 1.5, 1.5, 1.2, 1.2],
                "equilibrium": [u, cx * u, cy * u, 0, 0.5 * u],
            }
        ],
        "init": {u: (lambda x, y: np.sin(2 * np.pi * x) + np.cos(np.pi * y) + x * y)},
        "parameters": {LA: 1.0},
        "generator": generator,
    }
    return pylbm.Simulation(dico, sorder=sorder)


def advection_3d(generator, sorder):
    """D3Q7 advection on an anisotropic periodic domain."""
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.5], "z": [0.0, 0.75], "label": -1},
        "space_step": 0.125,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(7)),
                "conserved_moments": u,
                "polynomials": [
                    1,
                    X,
                    Y,
                    Z,
                    X**2 - Y**2,
                    X**2 - Z**2,
                    X**2 + Y**2 + Z**2,
                ],
                "relaxation_parameters": [0.0, 1.5, 1.5, 1.5, 1.2, 1.2, 1.2],
                "equilibrium": [u, 0.3 * u, -0.2 * u, 0.1 * u, 0, 0, 0.5 * u],
            }
        ],
        "init": {u: (lambda x, y, z: np.sin(2 * np.pi * x) + y * z + x)},
        "parameters": {LA: 1.0},
        "generator": generator,
    }
    return pylbm.Simulation(dico, sorder=sorder)


def run(build, generator, sorder, nsteps=5):
    sol = build(generator, sorder)
    for _ in range(nsteps):
        sol.one_time_step()
    return np.array(sol.m[u])


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("sorder", [list(p) for p in itertools.permutations(range(3))])
def test_storage_order_does_not_change_the_result_2d(generator, sorder):
    reference = run(advection_2d, "numpy", None)
    np.testing.assert_allclose(
        run(advection_2d, generator, sorder), reference, atol=1e-13
    )


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("sorder", [list(p) for p in itertools.permutations(range(4))])
def test_storage_order_does_not_change_the_result_3d(generator, sorder):
    reference = run(advection_3d, "numpy", None)
    np.testing.assert_allclose(
        run(advection_3d, generator, sorder), reference, atol=1e-13
    )
