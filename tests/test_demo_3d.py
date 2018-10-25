import importlib
import sys
import os
import pylbm
import pytest
import numpy as np

path = os.path.dirname(__file__) + '/../demo/3D'
path = os.path.abspath(path)


@pytest.fixture
def test_case_dir():
    sys.path.append(path)
    yield
    sys.path.pop()

@pytest.mark.slow
@pytest.mark.h5diff(single_reference=True)
@pytest.mark.usefixtures("test_case_dir")
@pytest.mark.parametrize('generator', ['numpy', 'cython'])
class Test1D:
    def runtest(self, dx, Tf, name, generator):
        module = importlib.import_module(name)
        return module.run(dx, Tf, generator=generator, withPlot=False)

    def test_advection(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'advection', generator)

    def test_karman(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Karman', generator)

    def test_lid_cavity(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'lid_cavity', generator)

    def test_poseuille(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'poiseuille', generator)        