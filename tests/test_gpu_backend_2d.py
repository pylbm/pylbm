import importlib
import os
import sympy as sp
import pytest
import numpy as np

path = os.path.dirname(__file__) + "/../demo/2D"
path = os.path.abspath(path)

U = sp.symbols("U")


@pytest.mark.gpu
class Test2DGpuBackend:
    def runtest(self, dx, Tf, name, generator):
        pytest.importorskip("loopy")
        pytest.importorskip("pyopencl")

        spec = importlib.util.spec_from_file_location(name, f"{path}/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.run(dx, Tf, generator=generator, with_plot=False)

    def test2D_advection_loopy_matches_cython(self):
        dx, Tf = 1.0 / 64, 0.5

        cython_sol = self.runtest(dx, Tf, "advection", "cython")
        loopy_sol = self.runtest(dx, Tf, "advection", "loopy")

        np.testing.assert_allclose(
            loopy_sol.m[U], cython_sol.m[U], rtol=1e-10, atol=1e-12
        )
