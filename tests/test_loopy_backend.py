"""
Loopy backend: same results as with Cython.
"""

import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y = sp.symbols("X, Y")
u = sp.symbols("u")

BOUNCE_BACK = [
    pylbm.bc.BounceBack,
    pylbm.bc.AntiBounceBack,
    pylbm.bc.BouzidiBounceBack,
    pylbm.bc.BouzidiAntiBounceBack,
]
NSTEPS = 20


def wall_value(f, m, x, y):
    m[u] = 1.0 + x


def time_value(f, m, t, x, y):
    m[u] = 1.0 + x * np.cos(t)


def simulation(generator, method, value, time_bc=False, obstacle=False):
    if generator == "loopy":
        pytest.importorskip("loopy")
        pytest.importorskip("pyopencl")

    boundary = {"method": {0: method}}
    if value is not None:
        boundary["value"] = value
        boundary["time_bc"] = time_bc

    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "label": [-1, -1, 0, 0]},
        "space_step": 1.0 / 16,
        "scheme_velocity": 1.0,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0.0, 1.5, 1.5, 1.2, 1.2],
                "equilibrium": [u, 0.1 * u, 0.2 * u, 0.0, 0.3 * u],
            }
        ],
        "init": {u: lambda x, y: 1.0 + 0.1 * np.sin(2 * np.pi * x)},
        "boundary_conditions": {0: boundary},
        "generator": generator,
    }
    if obstacle:
        # the wall cuts the links at distances other than 1/2
        dico["elements"] = [pylbm.Circle([0.5, 1.0], 0.23, label=0)]
    return pylbm.Simulation(dico)


def run(*args, **kwargs):
    sol = simulation(*args, **kwargs)
    for _ in range(NSTEPS):
        sol.one_time_step()
    return np.asarray(sol.m[u])


@pytest.mark.gpu
@pytest.mark.parametrize("method", BOUNCE_BACK)
@pytest.mark.parametrize(
    "value, time_bc",
    [(None, False), (wall_value, False), (time_value, True)],
    ids=["no value", "value", "time value"],
)
@pytest.mark.parametrize("obstacle", [False, True], ids=["walls", "obstacle"])
def test_boundary_conditions(method, value, time_bc, obstacle):
    expected = run("cython", method, value, time_bc, obstacle)
    result = run("loopy", method, value, time_bc, obstacle)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13)
    assert np.max(np.abs(expected - 1.0)) > 1e-2


@pytest.mark.gpu
def test_two_simulations_at_once():
    """
    Creating a second simulation does not make the arrays of the first one
    unusable.
    """
    methods = [pylbm.bc.BounceBack, pylbm.bc.AntiBounceBack]
    sols = [simulation("loopy", method, wall_value) for method in methods]
    for _ in range(NSTEPS):
        for sol in sols:
            sol.one_time_step()
    for sol, method in zip(sols, methods):
        np.testing.assert_allclose(
            np.asarray(sol.m[u]), run("cython", method, wall_value), rtol=0, atol=1e-13
        )
