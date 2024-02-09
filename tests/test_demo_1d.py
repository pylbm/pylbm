import importlib
import sys
import os
import pytest

path = os.path.dirname(__file__) + "/../demo/1D"
path = os.path.abspath(path)
dx = 1.0 / 64
Tf = 0.5
space_step = 1.0 / 64
final_time = 0.5


@pytest.mark.h5diff(single_reference=True)
@pytest.mark.parametrize("generator", ["numpy", "cython"])
class Test1D:
    def runtest(self, name, generator):
        spec = importlib.util.spec_from_file_location(name, f"{path}/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.run(dx, Tf, generator=generator, with_plot=False)

    def test1D_advection(self, generator):
        return self.runtest("advection", generator)

    def test1D_advection_reaction(self, generator):
        return self.runtest("advection_reaction", generator)

    def test1D_burgers(self, generator):
        return self.runtest("burgers", generator)

    def test1D_euler(self, generator):
        return self.runtest("euler", generator)


@pytest.fixture
def test1D_riemann_case_dir():
    sys.path.append(path + "/riemann_problems")
    yield
    sys.path.pop()


@pytest.mark.h5diff(single_reference=True)
@pytest.mark.usefixtures("test1D_riemann_case_dir")
@pytest.mark.parametrize("generator", ["numpy", "cython"])
class Test1DRiemann:
    def runtest(self, name, generator):
        spec = importlib.util.spec_from_file_location(
            name, f"{path}/riemann_problems/{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(space_step, final_time, generator=generator, with_plot=False)

    def test1D_riemann_advection(self, generator):
        return self.runtest("advection", generator)

    def test1D_riemann_burgers(self, generator):
        return self.runtest("burgers", generator)

    def test1D_riemann_euler(self, generator):
        return self.runtest("euler", generator)

    def test1D_riemann_euler_isothermal(self, generator):
        return self.runtest("euler_isothermal", generator)

    def test1D_riemann_p_system(self, generator):
        return self.runtest("p_system", generator)

    def test1D_riemann_shallow_water(self, generator):
        return self.runtest("shallow_water", generator)
