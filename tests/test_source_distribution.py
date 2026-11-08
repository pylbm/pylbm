"""
Source added to the distribution functions after the relaxation
(source_distribution).
"""

import numpy as np
import pytest
import sympy as sp

import pylbm

X, Y, Z, t = sp.symbols("X, Y, Z, t")
rho, qx, qy, u = sp.symbols("rho, qx, qy, u")

GENERATORS = ["numpy", "cython"]
WITH_LOOPY = [*GENERATORS, pytest.param("loopy", marks=pytest.mark.gpu)]

E = 3 * (X**2 + Y**2) - 4
EPS = sp.Rational(1, 2) * (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8)
QX = 3 * X * (X**2 + Y**2) - 5 * X
QY = 3 * Y * (X**2 + Y**2) - 5 * Y
D2Q9_BASIS = [1, X, Y, E, EPS, QX, QY, X**2 - Y**2, X * Y]
D2Q9_RATES = [0, 0, 0, 1.1, 1.2, 1.3, 1.3, 1.6, 1.6]
D2Q9_EQ = [
    rho,
    qx,
    qy,
    -2 * rho + 3 * (qx**2 + qy**2),
    rho - 3 * (qx**2 + qy**2),
    -qx,
    -qy,
    qx**2 - qy**2,
    qx * qy,
]

# velocities of the D2Q9 stencil, in the pylbm numbering
D2Q9_VX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
D2Q9_VY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
D2Q9_WEIGHTS = [sp.Rational(4, 9)] + [sp.Rational(1, 9)] * 4 + [sp.Rational(1, 36)] * 4


def force(fx, fy=0):
    """
    Body force (fx, fy) with the isotropic weights of the D2Q9:
    its momentum is (fx, fy) and its mass is 0.
    """
    return [
        3 * w * (vx * fx + vy * fy) for w, vx, vy in zip(D2Q9_WEIGHTS, D2Q9_VX, D2Q9_VY)
    ]


def d2q9(source=None, polynomials=D2Q9_BASIS, rates=D2Q9_RATES, eq=D2Q9_EQ, **kw):
    scheme = {
        "velocities": list(range(9)),
        "conserved_moments": [rho, qx, qy],
        "polynomials": polynomials,
        "relaxation_parameters": rates,
        "equilibrium": eq,
        **kw,
    }
    if source is not None:
        scheme["source_distribution"] = source
    return scheme


def periodic(schemes, generator, init=None, **kw):
    return {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": -1},
        "space_step": 1.0 / 8,
        "scheme_velocity": 1.0,
        "schemes": schemes,
        "init": init or {rho: 1.0, qx: 0.0, qy: 0.0},
        "generator": generator,
        **kw,
    }


def run(dico, nsteps):
    if dico["generator"] == "loopy":
        pytest.importorskip("loopy")
        pytest.importorskip("pyopencl")
    sol = pylbm.Simulation(dico)
    for _ in range(nsteps):
        sol.one_time_step()
    return sol


ALGORITHMS = [
    {},
    {"lbm_algorithm": {"settings": {"m_local": False}}},
    {"lbm_algorithm": {"settings": {"split": True}}},
    {"lbm_algorithm": {"name": pylbm.algorithm.BaseAlgorithm}},
]
GENERATOR_AND_ALGORITHM = [(g, a) for g in GENERATORS for a in ALGORITHMS]
GENERATOR_AND_ALGORITHM += [
    pytest.param("loopy", a, marks=pytest.mark.gpu) for a in ALGORITHMS
]


@pytest.mark.parametrize("generator, algorithm", GENERATOR_AND_ALGORITHM)
def test_constant_force(generator, algorithm):
    """
    A constant force adds its momentum to qx at each time step, whatever the
    algorithm that generates the time step.
    """
    nsteps, fx = 10, 1e-3
    sol = run(periodic([d2q9(force(fx))], generator, **algorithm), nsteps)
    np.testing.assert_allclose(sol.m[rho], 1.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qx], nsteps * fx, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qy], 0.0, rtol=0, atol=1e-13)


@pytest.mark.parametrize("generator", WITH_LOOPY)
def test_source_depending_on_time_and_on_a_conserved_moment(generator):
    """
    The source is evaluated with the conserved moments of the point and with
    the time at the beginning of the step.
    """
    nsteps, fx, rho0 = 10, 1e-3, 1.2
    init = {rho: rho0, qx: 0.0, qy: 0.0}
    dico = periodic([d2q9(force(fx * rho * t))], generator, init=init)
    sol = run(dico, nsteps)
    times = sol.dt * np.arange(nsteps)
    np.testing.assert_allclose(sol.m[rho], rho0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qx], fx * rho0 * times.sum(), rtol=0, atol=1e-13)


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("source, expected", [(X, 0), (Y, 1), (Z, 2)])
def test_source_depending_on_space(generator, source, expected):
    """
    A source that depends on a space variable sees that variable, Z included.
    The domain is anisotropic so that mixing up the coordinates changes the
    result.
    """
    dico = {
        "box": {"x": [0.0, 1.0], "y": [0.0, 2.0], "z": [0.0, 3.0], "label": -1},
        "space_step": 0.25,
        "scheme_velocity": 1.0,
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
                "source_distribution": [source, 0, 0, 0, 0, 0, 0],
            }
        ],
        "init": {u: 0.0},
        "generator": generator,
    }
    sol = run(dico, 1)
    coords = np.meshgrid(sol.domain.x, sol.domain.y, sol.domain.z, indexing="ij")
    np.testing.assert_allclose(sol.m[u], coords[expected], rtol=0, atol=1e-14)


@pytest.mark.parametrize("generator", GENERATORS)
def test_same_as_adding_the_source_before_the_boundary_conditions(generator):
    """
    Adding S after the relaxation is the same scheme as adding S to the
    populations before the boundary conditions of the next step, which is
    what the Stest=1 and Stest=-1 cases of I. Ginzburg do by hand: the
    populations differ by S at each step when the initial populations differ
    by S. Walls are included.
    """
    fx = 1e-3
    source = np.array(force(fx), dtype=float)

    def dico(with_source):
        return {
            "box": {"x": [0.0, 1.0], "y": [0.0, 0.5], "label": [-1, -1, 0, 0]},
            "space_step": 1.0 / 16,
            "scheme_velocity": 1.0,
            "schemes": [d2q9(force(fx) if with_source else None)],
            "init": {rho: 1.0, qx: 0.0, qy: 0.0},
            "boundary_conditions": {0: {"method": {0: pylbm.bc.BounceBack}}},
            "generator": generator,
        }

    def add_source(sol):
        for q, value in enumerate(source):
            sol.container.F[q] += value

    sol = pylbm.Simulation(dico(True))
    add_source(sol)
    ref = pylbm.Simulation(dico(False))
    for _ in range(100):
        sol.one_time_step()
        add_source(ref)
        ref.one_time_step()

    for q, value in enumerate(source):
        np.testing.assert_allclose(
            sol.container.F[q][1:-1, 1:-1],
            ref.container.F[q][1:-1, 1:-1] + value,
            rtol=0,
            atol=1e-14,
        )
    assert np.max(np.abs(ref.m[qx])) > 1e-2


@pytest.mark.parametrize("generator", GENERATORS)
def test_with_conserved_polynomials(generator):
    """
    A force on a TRT scheme written in a basis which contains neither 1, X
    nor Y gives the results of the same scheme in the usual basis.
    """
    s_plus, s_minus = 1.2, 1.7
    rates = [s_plus, s_minus, s_minus, s_plus, s_plus, s_minus, s_minus]
    rates += [s_plus, s_plus]
    usual_eq = [rho, qx, qy, -2 * rho, rho, -qx, -qy, 0, 0]
    usual = d2q9(force(1e-3, 2e-3), rates=rates, eq=usual_eq)

    basis = [1 + E, X + QX, Y + QY, *D2Q9_BASIS[3:]]
    eq = [
        usual_eq[0] + usual_eq[3],
        usual_eq[1] + usual_eq[5],
        usual_eq[2] + usual_eq[6],
        *usual_eq[3:],
    ]
    trt_b = d2q9(
        force(1e-3, 2e-3),
        polynomials=basis,
        rates=rates,
        eq=eq,
        conserved_polynomials=[1, X, Y],
    )

    ref = run(periodic([usual], generator), 20)
    sol = run(periodic([trt_b], generator), 20)
    for k in [rho, qx, qy]:
        np.testing.assert_allclose(sol.m[k], ref.m[k], rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qy], 20 * 2e-3, rtol=0, atol=1e-13)


@pytest.mark.parametrize("generator", GENERATORS)
def test_one_scheme_out_of_two(generator):
    """
    Only the second scheme has a source, which depends on a conserved moment
    of the first one.
    """
    ade = {
        "velocities": list(range(5)),
        "conserved_moments": u,
        "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
        "relaxation_parameters": [0.0, 1.4, 1.4, 1.1, 1.1],
        "equilibrium": [u, u * qx / rho, u * qy / rho, 0, 0.5 * u],
        "source_distribution": [1e-3 * rho, 0, 0, 0, 0],
    }
    init = {rho: 2.0, qx: 0.0, qy: 0.0, u: 0.0}
    sol = run(periodic([d2q9(), ade], generator, init=init), 5)
    np.testing.assert_allclose(sol.m[u], 5 * 2e-3, rtol=0, atol=1e-14)
    np.testing.assert_allclose(sol.m[qx], 0.0, rtol=0, atol=1e-14)


def scheme_dico(scheme):
    return {"dim": 2, "scheme_velocity": 1.0, "schemes": [scheme]}


def test_printed_with_the_scheme():
    fx = sp.Symbol("fx")
    dico = {**scheme_dico(d2q9(force(fx))), "parameters": {fx: 1e-3}}
    scheme = pylbm.Scheme(dico)
    assert "source distribution" in str(scheme)
    assert "source distribution" not in str(pylbm.Scheme(scheme_dico(d2q9())))


def test_wrong_number_of_values():
    with pytest.raises(ValueError, match="9 velocities but its 'source_distribution'"):
        pylbm.Scheme(scheme_dico(d2q9([0] * 8)))


def test_unknown_symbol():
    source = [0] * 8 + [sp.Symbol("m5")]
    with pytest.raises(ValueError, match="depends on m5"):
        pylbm.Scheme(scheme_dico(d2q9(source)))
