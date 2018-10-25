import importlib
import sys
import os
import pylbm
import pytest

path = os.path.dirname(__file__) + '/../demo/1D'
path = os.path.abspath(path)
dx = 1./64
Tf = 0.5

@pytest.fixture
def test_case_dir():
    sys.path.append(path)
    yield
    sys.path.pop()

@pytest.mark.h5diff(single_reference=True)
@pytest.mark.usefixtures("test_case_dir")
@pytest.mark.parametrize('generator', ['numpy', 'cython'])
class Test1D:
    def runtest(self, name, generator):
        module = importlib.import_module(name)
        return module.run(dx, Tf, generator=generator, withPlot=False)

    def test_advection(self, generator):
        return self.runtest('advection', generator)

    def test_advection_reaction(self, generator):
        return self.runtest('advection_reaction', generator)

    def test_burgers(self, generator):
        return self.runtest('burgers', generator)

    def test_euler(self, generator):
        return self.runtest('euler', generator)

    def test_p_system(self, generator):
        return self.runtest('p_system', generator)

    def test_shallow_water(self, generator):
        return self.runtest('shallow_water', generator)

