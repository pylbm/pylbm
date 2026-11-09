"""
Choice of the boundary condition link by link: a condition per velocity on
the same wall, a fallback condition for the links whose scheme reads a node
that is not a fluid node, and a warning when no fallback is given.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

import pylbm
from pylbm.boundary import BoundaryWarning

X, Y = sp.symbols("X, Y")
rho, qx, qy = sp.symbols("rho, qx, qy")
R2 = X**2 + Y**2
AXES = [1, 2, 3, 4]
DIAGONALS = [5, 6, 7, 8]


def d2q9_scheme():
    return {
        "velocities": list(range(9)),
        "conserved_moments": [rho, qx, qy],
        "polynomials": [
            1,
            X,
            Y,
            3 * R2 - 4,
            (9 * R2**2 - 21 * R2 + 8) / 2,
            3 * X * R2 - 5 * X,
            3 * Y * R2 - 5 * Y,
            X**2 - Y**2,
            X * Y,
        ],
        "relaxation_parameters": [0, 0, 0, 1.1, 1.2, 1.7, 1.7, 1.4, 1.4],
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
    }


def wall(f, m, x, y):
    m[rho] = 1.0
    m[qx] = 0.0
    m[qy] = 0.0


def lid(f, m, x, y):
    m[rho] = 1.0
    m[qx] = 0.05
    m[qy] = 0.0


def cavity(walls, lids=None, generator="numpy"):
    """
    Cavity with a moving lid, a disk (walls cut at any distance) and a
    notch one cell thick in the bottom wall.
    """
    lids = walls if lids is None else lids
    return {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": [0, 0, 0, 1]},
        "elements": [
            pylbm.Circle([0.5, 0.55], 0.17, label=0),
            pylbm.Parallelogram([0.3, 0.0], [1.0 / 32, 0], [0, 0.2], label=0),
        ],
        "space_step": 1.0 / 32,
        "scheme_velocity": 1.0,
        "schemes": [d2q9_scheme()],
        "init": {rho: 1.0, qx: 0.0, qy: 0.0},
        "boundary_conditions": {
            0: {"value": wall, **walls},
            1: {"value": lid, **lids},
        },
        "generator": generator,
    }


def links(sol):
    """Velocities of the links of each boundary class."""
    return {
        type(method).__name__: set(method.istore[:, 0].tolist())
        for method in sol.bc.methods
    }


def run(dico, steps=100):
    sol = pylbm.Simulation(dico)
    for _ in range(steps):
        sol.one_time_step()
    return np.array(sol.F_halo[:])[:, 1:-1, 1:-1]


def test_one_class_per_velocity():
    """The axis and the diagonal links of the same wall get their own class."""
    by_velocity = {
        "method": {
            0: {
                "default": pylbm.bc.CLIBounceBack,
                **{k: pylbm.bc.ParYLIBounceBack for k in DIAGONALS},
            }
        },
        "fallback": {0: pylbm.bc.BounceBack},
    }
    sol = pylbm.Simulation(cavity(by_velocity))
    found = links(sol)
    assert found["CLIBounceBack"] <= set(AXES)
    assert found["ParYLIBounceBack"] <= set(DIAGONALS)


def test_same_class_for_every_velocity():
    """A dictionary with one class for every velocity is that class."""
    plain = {
        "method": {0: pylbm.bc.CLIBounceBack},
        "fallback": {0: pylbm.bc.BounceBack},
    }
    by_velocity = {
        "method": {0: {k: pylbm.bc.CLIBounceBack for k in range(9)}},
        "fallback": {0: {"default": pylbm.bc.BounceBack}},
    }
    np.testing.assert_array_equal(run(cavity(plain)), run(cavity(by_velocity)))


def test_missing_velocity():
    dico = cavity({"method": {0: {k: pylbm.bc.CLIBounceBack for k in AXES}}})
    with pytest.raises(ValueError, match="no boundary method for the velocity 5"):
        pylbm.Simulation(dico)


def test_warning_without_fallback():
    """
    Next to the notch, links of CLI read the node behind them, which is in
    the solid: the simulation warns, once per label and class.
    """
    with pytest.warns(BoundaryWarning, match="label 0: .* links of CLIBounceBack"):
        sol = pylbm.Simulation(cavity({"method": {0: pylbm.bc.CLIBounceBack}}))
    assert sol.bc.missing_next_node[(0, "CLIBounceBack")] > 0


def test_fallback():
    """
    With a fallback, the links whose next node is not a fluid node use it,
    and the simulation does not warn.
    """
    dico = cavity(
        {"method": {0: pylbm.bc.CLIBounceBack}, "fallback": {0: pylbm.bc.BounceBack}}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        sol = pylbm.Simulation(dico)
    assert not sol.bc.missing_next_node

    # the fallback links are the ones that warned without fallback
    with pytest.warns(BoundaryWarning):
        reference = pylbm.Simulation(cavity({"method": {0: pylbm.bc.CLIBounceBack}}))
    fallback = [m for m in sol.bc.methods if isinstance(m, pylbm.bc.BounceBack)]
    assert sum(m.istore.shape[0] for m in fallback) == sum(
        reference.bc.missing_next_node.values()
    )

    # the links left to CLI that read the next node read a fluid node
    fluid = sol.bc.fluid_with_halo
    v = sol.scheme.stencil.get_all_velocities()
    for method in sol.bc.methods:
        if isinstance(method, pylbm.bc.CLIBounceBack):
            reads = method.needs_next_node(method.distance)
            k = method.istore[reads, 0]
            next_node = method.istore[reads, 1:] + 2 * v[k]
            assert fluid[tuple(next_node.T)].all()


def test_fallback_must_not_need_the_next_node():
    dico = cavity(
        {
            "method": {0: pylbm.bc.CLIBounceBack},
            "fallback": {0: pylbm.bc.YLIBounceBack},
        }
    )
    with pytest.raises(ValueError, match="the fallback YLIBounceBack needs"):
        pylbm.Simulation(dico)


@pytest.mark.parametrize("generator", ["numpy", "cython"])
def test_periodic_faces_are_fluid(generator):
    """
    In a channel periodic in x, the next node of the links next to the
    periodic faces is a copy of a fluid node: no warning.
    """
    delta = 0.3
    n = 6
    y0, y1 = 0.5 - delta, n - 0.5 + delta
    dico = {
        "box": {"x": [0.0, 4.0], "y": [-1.0, n + 1.0], "label": [-1, -1, 0, 0]},
        "elements": [
            pylbm.Parallelogram((-1.0, y0 - 2.0), (6.0, 0.0), (0.0, 2.0), label=0),
            pylbm.Parallelogram((-1.0, y1), (6.0, 0.0), (0.0, 2.0), label=0),
        ],
        "space_step": 1.0,
        "scheme_velocity": 1.0,
        "schemes": [d2q9_scheme()],
        "init": {rho: 1.0, qx: 0.0, qy: 0.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.ParYLIBounceBack}, "value": wall}
        },
        "generator": generator,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        sol = pylbm.Simulation(dico)
    assert not sol.bc.missing_next_node


def test_numpy_and_cython_with_fallback():
    walls = {
        "method": {
            0: {
                "default": pylbm.bc.CLIBounceBack,
                **{k: pylbm.bc.ParYLIBounceBack for k in DIAGONALS},
            }
        },
        "fallback": {0: pylbm.bc.BounceBack},
    }
    np.testing.assert_allclose(
        run(cavity(walls, generator="numpy")),
        run(cavity(walls, generator="cython")),
        rtol=0,
        atol=1e-14,
    )


def test_validation():
    dico = cavity({"method": {0: {"diagonals": pylbm.bc.CLIBounceBack}}})
    with pytest.raises(SystemExit):
        pylbm.Simulation(dico)
