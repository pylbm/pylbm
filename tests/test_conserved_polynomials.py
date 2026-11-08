"""
Conserved moments defined with their own polynomials (conserved_polynomials).
"""

import numpy as np
import pytest
import sympy as sp

import pylbm
from pylbm.symbolic import recursive_sub

X, Y, LA = sp.symbols("X, Y, lambda")
rho, qx, qy, u = sp.symbols("rho, qx, qy, u")

GENERATORS = ["numpy", "cython"]

E = 3 * (X**2 + Y**2) - 4
EPS = sp.Rational(1, 2) * (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8)
QX = 3 * X * (X**2 + Y**2) - 5 * X
QY = 3 * Y * (X**2 + Y**2) - 5 * Y
D2Q9_BASIS = [1, X, Y, E, EPS, QX, QY, X**2 - Y**2, X * Y]


def d2q9(polynomials, relaxation, equilibrium, conserved_polynomials=None):
    scheme = {
        "velocities": list(range(9)),
        "polynomials": polynomials,
        "relaxation_parameters": relaxation,
        "equilibrium": equilibrium,
        "conserved_moments": [rho, qx, qy],
    }
    if conserved_polynomials is not None:
        scheme["conserved_polynomials"] = conserved_polynomials
    return scheme


def mrt_d2q9(conserved_polynomials=None):
    return d2q9(
        D2Q9_BASIS,
        [0, 0, 0, 1.1, 1.2, 1.3, 1.3, 1.6, 1.6],
        [
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
        conserved_polynomials,
    )


def periodic(schemes, init, generator):
    return {
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": -1},
        "space_step": 1.0 / 16,
        "scheme_velocity": 1.0,
        "schemes": schemes,
        "init": init,
        "generator": generator,
    }


def wave_init():
    return {
        rho: lambda x, y: 1.0 + 0.1 * np.sin(2 * np.pi * x),
        qx: lambda x, y: 0.05 * np.cos(2 * np.pi * y),
        qy: 0.0,
    }


def run(dico, nsteps):
    sol = pylbm.Simulation(dico)
    for _ in range(nsteps):
        sol.one_time_step()
    return sol


@pytest.mark.parametrize("generator", GENERATORS)
def test_same_as_basis_rows(generator):
    """
    When the conserved polynomials are rows of the basis, the results are
    the ones of the scheme without conserved_polynomials, walls included.
    """

    def wall(f, m, x, y):
        m[rho] = 1.0
        m[qx] = 0.0
        m[qy] = 0.0

    def lid(f, m, x, y):
        m[rho] = 1.0
        m[qx] = 0.05
        m[qy] = 0.0

    def dico(conserved_polynomials):
        return {
            "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": [0, 0, 0, 1]},
            "space_step": 1.0 / 16,
            "scheme_velocity": LA,
            "parameters": {LA: 1.0},
            "schemes": [mrt_d2q9(conserved_polynomials)],
            "init": {rho: 1.0, qx: 0.0, qy: 0.0},
            "boundary_conditions": {
                0: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": wall},
                1: {"method": {0: pylbm.bc.BouzidiBounceBack}, "value": lid},
            },
            "generator": generator,
        }

    ref = run(dico(None), 100)
    sol = run(dico([1, X, Y]), 100)

    for k in [rho, qx, qy]:
        np.testing.assert_allclose(sol.m[k], ref.m[k], rtol=0, atol=1e-13)
    assert np.max(np.abs(ref.m[qx])) > 1e-3
    for k in range(9):
        np.testing.assert_allclose(sol.m[k], ref.m[k], rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.F[0], ref.F[0], rtol=0, atol=1e-13)


@pytest.mark.parametrize("generator", GENERATORS)
def test_conserved_moments_come_first(generator):
    """
    As without conserved_polynomials, m[i] is the i-th conserved moment.
    """
    sol = run(periodic([mrt_d2q9([1, X, Y])], wave_init(), generator), 3)
    assert sol.scheme.nmoments == 12
    assert list(sol.scheme.consm.values()) == [0, 1, 2]
    for i, k in enumerate([rho, qx, qy]):
        np.testing.assert_array_equal(sol.m[i], sol.m[k])


@pytest.mark.parametrize("generator", GENERATORS)
def test_trt_basis_without_conserved_rows(generator):
    """
    A TRT scheme written in a basis which contains neither 1, X nor Y
    (the TRT_b basis of the release 10/2026 of I. Ginzburg) gives the
    results of the same TRT scheme in the usual basis.
    """
    s_plus, s_minus = 1.2, 1.7
    trt_rates = [s_plus, s_minus, s_minus, s_plus, s_plus, s_minus, s_minus]
    trt_rates += [s_plus, s_plus]
    usual_eq = [rho, qx, qy, -2 * rho, rho, -qx, -qy, 0, 0]
    usual = d2q9(D2Q9_BASIS, trt_rates, usual_eq)

    # 1 + E, X + QX and Y + QY replace 1, X and Y: symmetric and
    # antisymmetric vectors relax with the same rate, so the scheme is the
    # same, but its conserved moments are no longer rows of the basis
    basis = [1 + E, X + QX, Y + QY, *D2Q9_BASIS[3:]]
    eq = [
        usual_eq[0] + usual_eq[3],
        usual_eq[1] + usual_eq[5],
        usual_eq[2] + usual_eq[6],
        *usual_eq[3:],
    ]
    trt_b = d2q9(basis, trt_rates, eq, conserved_polynomials=[1, X, Y])

    ref = run(periodic([usual], wave_init(), generator), 50)
    sol = run(periodic([trt_b], wave_init(), generator), 50)

    for k in [rho, qx, qy]:
        np.testing.assert_allclose(sol.m[k], ref.m[k], rtol=0, atol=1e-13)
    assert np.max(np.abs(ref.m[qx] - ref.m[qx][0, 0])) > 1e-3


@pytest.mark.parametrize("generator", GENERATORS)
def test_source_in_the_equilibrium(generator):
    """
    A force F put in the equilibrium of the momentum rows,
    qx + F / s, adds F to qx at each time step.
    """
    force, s_minus = 1e-3, 1.7
    eq = [rho, qx + force / s_minus, qy, -2 * rho, rho]
    eq += [-(qx + force / s_minus), -qy, 0, 0]
    rates = [1.2, s_minus, s_minus, 1.2, 1.2, s_minus, s_minus, 1.2, 1.2]
    scheme = d2q9(D2Q9_BASIS, rates, eq, conserved_polynomials=[1, X, Y])

    # the populations are initialized with the equilibrium: start from
    # qx = -F / s so that their momentum is 0
    init = {rho: 1.0, qx: -force / s_minus, qy: 0.0}
    sol = pylbm.Simulation(periodic([scheme], init, generator))
    np.testing.assert_allclose(sol.m[qx], 0.0, rtol=0, atol=1e-15)
    nsteps = 20
    for _ in range(nsteps):
        sol.one_time_step()

    np.testing.assert_allclose(sol.m[rho], 1.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qx], nsteps * force, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sol.m[qy], 0.0, rtol=0, atol=1e-13)


@pytest.mark.parametrize("generator", GENERATORS)
def test_one_scheme_out_of_two(generator):
    """
    Only the second scheme (advection-diffusion of u by the flow) uses
    conserved_polynomials.
    """

    def ade(conserved_polynomials):
        scheme = {
            "velocities": list(range(5)),
            "conserved_moments": u,
            "polynomials": [1, X, Y, X**2 - Y**2, X**2 + Y**2],
            "relaxation_parameters": [0.0, 1.4, 1.4, 1.1, 1.1],
            "equilibrium": [u, u * qx / rho, u * qy / rho, 0, 0.5 * u],
        }
        if conserved_polynomials is not None:
            scheme["conserved_polynomials"] = conserved_polynomials
        return scheme

    init = {**wave_init(), u: lambda x, y: np.exp(-20 * ((x - 0.5) ** 2))}
    ref = run(periodic([mrt_d2q9(), ade(None)], init, generator), 30)
    sol = run(periodic([mrt_d2q9(), ade(1)], init, generator), 30)

    assert sol.scheme.nmoments == 15
    assert list(sol.scheme.consm.values()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(sol.m[3], sol.m[u])
    for k in [rho, qx, qy, u]:
        np.testing.assert_allclose(sol.m[k], ref.m[k], rtol=0, atol=1e-13)


def test_stability_matrix():
    """
    The amplification matrix does not depend on the way the conserved
    moments are defined.
    """

    def relaxation_matrix(conserved_polynomials):
        dico = {
            "dim": 2,
            "scheme_velocity": LA,
            "parameters": {LA: 1.0},
            "schemes": [mrt_d2q9(conserved_polynomials)],
        }
        stability = pylbm.Stability(pylbm.Scheme(dico))
        values = list(zip(stability.consm, [1.0, 0.05, 0.02])) + [(LA, 1.0)]
        matrix = recursive_sub(stability.relax_mat_f, values)
        return np.asarray(matrix).astype(float)

    np.testing.assert_allclose(
        relaxation_matrix([1, LA * X, LA * Y]),
        relaxation_matrix(None),
        rtol=0,
        atol=1e-14,
    )


def scheme_dico(scheme, **kwargs):
    return {"dim": 2, "scheme_velocity": 1.0, "schemes": [scheme], **kwargs}


def test_conserved_moment_missing_from_the_equilibrium():
    scheme = mrt_d2q9()
    scheme["equilibrium"] = [rho, qx + 0.1, *scheme["equilibrium"][2:]]
    with pytest.raises(ValueError, match="qx of the scheme 0 is not a row"):
        pylbm.Scheme(scheme_dico(scheme))
    scheme["conserved_polynomials"] = [1, X, Y]
    assert pylbm.Scheme(scheme_dico(scheme)).nmoments == 12


def test_wrong_number_of_conserved_polynomials():
    with pytest.raises(ValueError, match="3 conserved moments but 2"):
        pylbm.Scheme(scheme_dico(mrt_d2q9([1, X])))


def test_source_terms_on_a_conserved_polynomial():
    scheme = mrt_d2q9([1, X, Y])
    scheme["source_terms"] = {qx: 0.1}
    with pytest.raises(NotImplementedError, match="acts on qx"):
        pylbm.Scheme(scheme_dico(scheme))


def test_relative_velocity():
    dico = scheme_dico(mrt_d2q9([1, X, Y]), relative_velocity=[qx / rho, qy / rho])
    with pytest.raises(NotImplementedError, match="relative_velocity"):
        pylbm.Scheme(dico)


def test_equivalent_equation():
    with pytest.raises(NotImplementedError, match="conserved_polynomials"):
        pylbm.EquivalentEquation(pylbm.Scheme(scheme_dico(mrt_d2q9([1, X, Y]))))
