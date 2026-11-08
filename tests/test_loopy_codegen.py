"""
Code generated for the Loopy backend.
"""

import os
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

SCRIPT = """
import sys
import sympy as sp
import pylbm

X, Y, u = sp.symbols("X, Y, u")
sorder = [int(i) for i in sys.argv[1].split(",")]
dico = {
    "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": -1},
    "space_step": 0.125,
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
    "init": {u: 1.0},
    "generator": "loopy",
    "show_code": True,
}
pylbm.Simulation(dico, sorder=sorder)
"""


def generated_code(sorder, hash_seed):
    pytest.importorskip("loopy")
    pytest.importorskip("pyopencl")
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, ",".join(map(str, sorder))],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def first_axis(code, kernel):
    """Return the index of the kernel mapped to the first work group axis."""
    pattern = rf'{kernel} = lp.split_iname\({kernel}, "(\w+)", \d+, outer_tag="g.0"'
    return re.search(pattern, code).group(1)


def test_same_code_whatever_the_hash_seed():
    codes = {generated_code([0, 1, 2], seed) for seed in range(4)}
    assert len(codes) == 1


@pytest.mark.parametrize(
    "sorder, contiguous", [([0, 1, 2], "iy__"), ([0, 2, 1], "ix__")]
)
def test_contiguous_index_on_the_first_axis(sorder, contiguous):
    """
    The last index of the storage order is the most contiguous in memory: it
    goes to the first axis of the work groups.
    """
    code = generated_code(sorder, 0)
    for kernel in ["transport", "f2m", "one_time_step"]:
        assert first_axis(code, kernel) == contiguous
