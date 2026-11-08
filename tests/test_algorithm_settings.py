"""
The settings of the algorithm change the generated code, not the results.
"""

import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y = sp.symbols("X, Y")
rho, qx, qy = sp.symbols("rho, qx, qy")

GENERATORS = ["numpy", "cython", pytest.param("loopy", marks=pytest.mark.gpu)]
SETTINGS = [
    {"m_local": False},
    {"m_local": True},
    {"split": True},
    {"m_local": True, "split": True},
]


def run(generator, settings):
    if generator == "loopy":
        pytest.importorskip("loopy")
        pytest.importorskip("pyopencl")

    E = 3 * (X**2 + Y**2) - 4
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": -1},
        "space_step": 1.0 / 16,
        "scheme_velocity": 1.0,
        "schemes": [
            {
                "velocities": list(range(9)),
                "conserved_moments": [rho, qx, qy],
                "polynomials": [
                    1,
                    X,
                    Y,
                    E,
                    (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8) / 2,
                    3 * X * (X**2 + Y**2) - 5 * X,
                    3 * Y * (X**2 + Y**2) - 5 * Y,
                    X**2 - Y**2,
                    X * Y,
                ],
                "relaxation_parameters": [0, 0, 0, 1.1, 1.2, 1.3, 1.3, 1.6, 1.6],
                "equilibrium": [
                    rho,
                    qx,
                    qy,
                    -2 * rho + 3 * (qx**2 + qy**2),
                    rho - 3 * (qx**2 + qy**2),
                    -qx,
                    -qy,
                    qx**2 - qy**2,
                    qx * qy,
                ],
                "source_terms": {qx: 1e-3 * rho},
            }
        ],
        "init": {
            rho: 1.0,
            qx: lambda x, y: 0.01 * np.sin(2 * np.pi * y),
            qy: 0.0,
        },
        "generator": generator,
    }
    if settings is not None:
        dico["lbm_algorithm"] = {"settings": settings}

    sol = pylbm.Simulation(dico)
    for _ in range(30):
        sol.one_time_step()
    return np.array([np.asarray(sol.m[k]) for k in (rho, qx, qy)])


@pytest.mark.parametrize("settings", SETTINGS, ids=str)
@pytest.mark.parametrize("generator", GENERATORS)
def test_same_results(generator, settings):
    expected = run(generator, None)
    np.testing.assert_allclose(run(generator, settings), expected, rtol=0, atol=1e-14)
    assert np.max(np.abs(expected[1])) > 1e-3


def run_with_obstacle(generator, check_isfluid):
    if generator == "loopy":
        pytest.importorskip("loopy")
        pytest.importorskip("pyopencl")

    u = sp.symbols("u")
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": -1},
        "elements": [pylbm.Circle([0.5, 0.5], 0.2, label=0)],
        "space_step": 1.0 / 32,
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
        "boundary_conditions": {0: {"method": {0: pylbm.bc.Neumann}}},
        "lbm_algorithm": {"settings": {"check_isfluid": check_isfluid}},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    for _ in range(20):
        sol.one_time_step()
    fluid = sol.domain.in_or_out == sol.domain.valin
    inner = tuple(slice(1, -1) for _ in range(2))
    values = sol.m[u]
    # Loopy gives a device array
    values = values.get() if hasattr(values, "get") else np.asarray(values)
    return values[fluid[inner]]


@pytest.mark.parametrize("generator", GENERATORS)
def test_check_isfluid(generator):
    """
    Skipping the solid points does not change the fluid points.
    """
    expected = run_with_obstacle(generator, False)
    result = run_with_obstacle(generator, True)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
    assert np.max(np.abs(expected - 1.0)) > 1e-2
