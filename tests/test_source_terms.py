import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y, Z, LA = sp.symbols("X, Y, Z, lambda")
u = sp.symbols("u")


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize(
    "source, expected",
    [
        (X, lambda x, y, z: x),
        (Y, lambda x, y, z: y),
        (Z, lambda x, y, z: z),
    ],
)
def test_source_term_uses_the_right_coordinate_in_3d(generator, source, expected):
    """
    A source term that depends on a space variable must see that variable,
    including Z in 3D. The domain is anisotropic so that mixing up the
    coordinates changes the result.
    """
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "z": [0.0, 3.0], "label": -1},
        "space_step": 0.25,
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
                "relaxation_parameters": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "equilibrium": [u, 0, 0, 0, 0, 0, 0],
                "source_terms": {u: source},
            }
        ],
        "init": {u: 0.0},
        "parameters": {LA: 1.0},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    sol.one_time_step()

    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # u is constant in space before the step and the source does not depend
    # on u, so one step adds exactly dt * source(x, y, z).
    np.testing.assert_allclose(sol.m[u], sol.dt * expected(xx, yy, zz), atol=1e-14)


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("sorder", [None, [2, 0, 1], [1, 2, 0]])
def test_equilibrium_and_source_depending_on_space(generator, sorder):
    """
    An equilibrium that depends on the space variables must be evaluated at
    the right point, whatever the storage order of the arrays.
    """
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "label": -1},
        "space_step": 0.25,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0.0, 1.0, 1.0, 1.0, 1.0],
                "equilibrium": [u, X * u, Y * u, 0, 0],
                "source_terms": {u: X + 2 * Y},
            }
        ],
        "init": {u: 1.0},
        "parameters": {LA: 1.0},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico, sorder=sorder)
    xx, yy = np.meshgrid(sol.domain.x, sol.domain.y, indexing="ij")

    # initialization through the equilibrium
    np.testing.assert_allclose(sol.m[1], xx, atol=1e-14)
    np.testing.assert_allclose(sol.m[2], yy, atol=1e-14)

    # expected result of the pull transport on the periodic domain
    velocities = sol.scheme.stencil.get_all_velocities()
    u_transport = sum(
        np.roll(np.asarray(sol.F[q]), shift=tuple(v), axis=(0, 1))
        for q, v in enumerate(velocities)
    )
    source = xx + 2 * yy

    sol.one_time_step()

    # source with dt/2, relaxation with s = 1 towards the equilibrium,
    # then source with dt/2
    u_half = u_transport + 0.5 * sol.dt * source
    np.testing.assert_allclose(sol.m[u], u_half + 0.5 * sol.dt * source, atol=1e-13)
    np.testing.assert_allclose(sol.m[1], xx * u_half, atol=1e-13)
    np.testing.assert_allclose(sol.m[2], yy * u_half, atol=1e-13)


@pytest.mark.parametrize("generator", ["numpy", "cython"])
def test_space_dependent_scheme_with_boundary_values(generator):
    """
    The boundary conditions compute their equilibrium on arrays of boundary
    points. This must keep working when the bulk scheme depends on space.
    """

    def wall_value(f, m, x, y):
        m[u] = 1.0

    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "label": [-1, -1, 0, 0]},
        "space_step": 0.25,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0.0, 1.0, 1.0, 1.0, 1.0],
                "equilibrium": [u, 0, 0, 0, 0],
                "source_terms": {u: X},
            }
        ],
        "init": {u: 1.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.AntiBounceBack}, "value": wall_value}
        },
        "parameters": {LA: 1.0},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    sol.one_time_step()
    assert np.all(np.isfinite(sol.m[u]))
