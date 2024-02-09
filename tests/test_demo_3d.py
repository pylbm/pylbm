import importlib
import os
import pytest

path = os.path.dirname(__file__) + "/../demo/3D"
path = os.path.abspath(path)


@pytest.mark.slow
@pytest.mark.h5diff(single_reference=True)
@pytest.mark.parametrize("generator", ["numpy", "cython"])
class Test3D:
    def runtest(self, dx, Tf, name, generator):
        spec = importlib.util.spec_from_file_location(name, f"{path}/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(dx, Tf, generator=generator, with_plot=False)

    def test3D_advection(self, generator):
        dx, Tf = 1.0 / 64, 0.5
        return self.runtest(dx, Tf, "advection", generator)

    def test3D_karman(self, generator):
        dx, Tf = 1.0 / 64, 0.5
        return self.runtest(dx, Tf, "Karman", generator)

    def test3D_lid_cavity(self, generator):
        dx, Tf = 1.0 / 64, 0.5
        return self.runtest(dx, Tf, "lid_cavity", generator)

    def test3D_poseuille(self, generator):
        dx, Tf = 1.0 / 64, 0.5
        return self.runtest(dx, Tf, "poiseuille", generator)
