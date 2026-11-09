"""
Linear interpolations of the bounce-back and of the anti-bounce-back
(LinearBounceBack, LinearAntiBounceBack and their subclasses).

The parametrized schemes give the exact steady solution of a force-driven
Poiseuille flow, or of a diffusion with a constant source, between two walls
at a distance delta of the fluid nodes, for any relaxation rate, when the
product of the TRT parameters is the one of the scheme.
"""

import numpy as np
import pytest
import sympy as sp

import pylbm
from pylbm.boundary import LinearAntiBounceBack, LinearBounceBack

X, Y = sp.symbols("X, Y")
rho, qx, qy, u = sp.symbols("rho, qx, qy, u")

# number of fluid nodes between the walls
N = 8


def channel(delta):
    """
    Box and walls of a channel periodic in x with N fluid nodes in y, at
    y = 0.5, ..., N - 0.5, and walls at a distance delta of the first and
    the last ones.
    """
    y0, y1 = 0.5 - delta, N - 0.5 + delta
    geometry = {
        "box": {"x": [0.0, 2.0], "y": [-1.0, N + 1.0], "label": [-1, -1, 0, 0]},
        "elements": [
            pylbm.Parallelogram((-1.0, y0 - 2.0), (4.0, 0.0), (0.0, 2.0), label=0),
            pylbm.Parallelogram((-1.0, y1), (4.0, 0.0), (0.0, 2.0), label=0),
        ],
        "space_step": 1.0,
        "scheme_velocity": 1.0,
    }
    return geometry, y0, y1


def run_to_steady_state(sol, moment):
    previous = None
    for _ in range(200):
        for _ in range(200):
            sol.one_time_step()
        current = np.array(sol.m[moment])[0]
        if previous is not None and np.max(np.abs(current - previous)) < 5e-14 * np.max(
            np.abs(current)
        ):
            return current
        previous = current
    raise RuntimeError("no steady state")


def poiseuille(method, delta, nu, generator, force=1e-3):
    """
    Force-driven Poiseuille flow with a d2Q9 TRT scheme and a force added
    after the collision with the isotropic weights. The scheme is exact when
    Lambda+ Lambda- = delta^2 / 2 and the wall momentum is Lambda- F.
    """
    lam_p = 3 * nu
    lam_m = 0.5 * delta**2 / lam_p
    s_p, s_m = 1 / (0.5 + lam_p), 1 / (0.5 + lam_m)
    weights = {0: 4 / 9, 1: 1 / 9, 2: 1 / 36}
    stencil = pylbm.Stencil({"dim": 2, "schemes": [{"velocities": list(range(9))}]})
    source = [
        3 * weights[abs(c[0]) + abs(c[1])] * c[0] * force
        for c in stencil.get_all_velocities()
    ]

    def wall(f, m, x, y):
        m[rho] = 1.0
        m[qx] = lam_m * force
        m[qy] = 0.0

    geometry, y0, y1 = channel(delta)
    r2 = X**2 + Y**2
    dico = {
        **geometry,
        "schemes": [
            {
                "velocities": list(range(9)),
                "conserved_moments": [rho, qx, qy],
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
                "relaxation_parameters": [0, 0, 0, s_p, s_p, s_m, s_m, s_p, s_p],
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
                "source_distribution": source,
            }
        ],
        "init": {rho: 1.0, qx: 0.0, qy: 0.0},
        "boundary_conditions": {0: {"method": {0: method}, "value": wall}},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    # the stored populations hold the force: the momentum is the velocity
    # plus F/2
    ux = run_to_steady_state(sol, qx) - 0.5 * force
    y = sol.domain.y
    fluid = (y > y0) & (y < y1)
    exact = force / (2 * nu) * (y[fluid] - y0) * (y1 - y[fluid])
    return ux[fluid], exact


def diffusion(method, delta, lam_p, generator, source=1e-3, t=1 / 6):
    """
    Diffusion with a constant source, with a d2Q5 TRT scheme and a source
    added after the collision with the equilibrium weights. The scheme is
    exact when Lambda+ Lambda- = delta^2 / (2 (1 - 2 t)).
    """
    lam_m = 0.5 * delta**2 / (1 - 2 * t) / lam_p
    s_p, s_m = 1 / (0.5 + lam_p), 1 / (0.5 + lam_m)
    diffusivity = 2 * t * lam_m

    def wall(f, m, x, y):
        m[u] = 0.0

    geometry, y0, y1 = channel(delta)
    dico = {
        **geometry,
        "schemes": [
            {
                "velocities": list(range(5)),
                "conserved_moments": u,
                "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
                "relaxation_parameters": [0, s_m, s_m, s_p, s_p],
                "equilibrium": [u, 0, 0, 0, 4 * t * u],
                "source_distribution": [(1 - 4 * t) * source] + [t * source] * 4,
            }
        ],
        "init": {u: 0.0},
        "boundary_conditions": {0: {"method": {0: method}, "value": wall}},
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    c = run_to_steady_state(sol, u) - 0.5 * source
    y = sol.domain.y
    fluid = (y > y0) & (y < y1)
    exact = source / (2 * diffusivity) * (y[fluid] - y0) * (y1 - y[fluid])
    return c[fluid], exact


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("nu", [0.05, 0.3])
@pytest.mark.parametrize("delta", [0.125, 0.625])
@pytest.mark.parametrize(
    "method",
    [pylbm.bc.CLIBounceBack, pylbm.bc.ParYLIBounceBack, pylbm.bc.ParBFLBounceBack],
)
def test_parametrized_bounce_back_poiseuille(method, delta, nu, generator):
    ux, exact = poiseuille(method, delta, nu, generator)
    np.testing.assert_allclose(ux, exact, rtol=0, atol=1e-12 * exact.max())


@pytest.mark.parametrize("generator", ["numpy", "cython"])
@pytest.mark.parametrize("lam_p", [0.2, 0.5])
@pytest.mark.parametrize("delta", [0.375, 0.625])
@pytest.mark.parametrize(
    "method", [pylbm.bc.LMKCAntiBounceBack, pylbm.bc.ParLMKAAntiBounceBack]
)
def test_parametrized_anti_bounce_back_diffusion(method, delta, lam_p, generator):
    c, exact = diffusion(method, delta, lam_p, generator)
    np.testing.assert_allclose(c, exact, rtol=0, atol=1e-11 * exact.max())


def test_yli_without_correction_is_not_exact():
    """The correction is what makes YLI exact."""
    ux, exact = poiseuille(pylbm.bc.YLIBounceBack, 0.375, 0.3, "numpy")
    assert np.max(np.abs(ux - exact)) > 1e-3 * exact.max()


def cavity(method, generator, steps):
    """
    Lid-driven cavity with a square obstacle and a notch in the bottom wall
    (convex and concave corners), and a body force.
    """
    r2 = X**2 + Y**2
    force = (1e-4, -2e-4)
    weights = {0: 4 / 9, 1: 1 / 9, 2: 1 / 36}
    stencil = pylbm.Stencil({"dim": 2, "schemes": [{"velocities": list(range(9))}]})
    source = [
        3 * weights[abs(c[0]) + abs(c[1])] * (c[0] * force[0] + c[1] * force[1])
        for c in stencil.get_all_velocities()
    ]

    def lid(f, m, x, y):
        m[rho] = 1.0
        m[qx] = 0.05
        m[qy] = 0.0

    def wall(f, m, x, y):
        m[rho] = 1.0
        m[qx] = 0.0
        m[qy] = 0.0

    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": [0, 0, 0, 1]},
        "elements": [
            pylbm.Parallelogram([0.4, 0.4], [0.23, 0], [0, 0.17], label=0),
            pylbm.Parallelogram([0.0, 0.0], [0.3, 0], [0, 0.21], label=0),
        ],
        "space_step": 1.0 / 32,
        "scheme_velocity": 1.0,
        "schemes": [
            {
                "velocities": list(range(9)),
                "conserved_moments": [rho, qx, qy],
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
                "source_distribution": source,
            }
        ],
        "init": {rho: 1.0, qx: 0.0, qy: 0.0},
        "boundary_conditions": {
            0: {"method": {0: method}, "value": wall},
            1: {"method": {0: method}, "value": lid},
        },
        "generator": generator,
    }
    sol = pylbm.Simulation(dico)
    for _ in range(steps):
        sol.one_time_step()
    return np.array(sol.F_halo[:])[:, 1:-1, 1:-1]


@pytest.mark.parametrize(
    "method", [pylbm.bc.ParYLIBounceBack, pylbm.bc.ParLMKAAntiBounceBack]
)
def test_correction_numpy_and_cython(method):
    """
    The correction reads the populations of the previous time step: NumPy
    then stores them in a second array, and the two backends give the same
    populations, corners and first time step included.
    """
    np.testing.assert_allclose(
        cavity(method, "numpy", 50), cavity(method, "cython", 50), rtol=0, atol=1e-14
    )


class _BFL(LinearBounceBack):
    def alpha(self, delta):
        return np.where(delta <= 0.5, 2.0, 1.0 / np.maximum(delta, 0.5))


class _LMKA(LinearAntiBounceBack):
    def alpha(self, delta):
        return -np.where(delta <= 0.5, 2.0, 1.0 / np.maximum(delta, 0.5))


@pytest.mark.parametrize(
    "generic, method, generators",
    [
        (_BFL, pylbm.bc.BouzidiBounceBack, ["numpy", "cython"]),
        (_LMKA, pylbm.bc.BouzidiAntiBounceBack, ["numpy"]),
    ],
)
def test_generic_scheme_gives_bouzidi(generic, method, generators):
    """
    With the coefficients of Bouzidi and without correction, the generic
    scheme gives the Bouzidi conditions.
    """
    for generator in generators:
        np.testing.assert_allclose(
            cavity(generic, generator, 50),
            cavity(method, generator, 50),
            rtol=0,
            atol=1e-14,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "method", [pylbm.bc.ParYLIBounceBack, pylbm.bc.ParLMKAAntiBounceBack]
)
def test_correction_loopy(method):
    np.testing.assert_allclose(
        cavity(method, "loopy", 50), cavity(method, "numpy", 50), rtol=0, atol=1e-14
    )
