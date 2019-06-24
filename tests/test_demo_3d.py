import importlib
import sys
import os
import pylbm
import pytest
import numpy as np

path = os.path.dirname(__file__) + '/../demo/3D'
path = os.path.abspath(path)


@pytest.fixture
def test3D_case_dir():
    sys.path.append(path)
    yield
    sys.path.pop()

@pytest.mark.slow
@pytest.mark.h5diff(single_reference=True)
@pytest.mark.usefixtures("test3D_case_dir")
@pytest.mark.parametrize('generator', ['numpy', 'cython'])
class Test3D:
    def runtest(self, dx, Tf, name, generator):
        module = importlib.import_module(name)
        return module.run(dx, Tf, generator=generator, with_plot=False)

    def test3D_advection(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'advection', generator)

    def test3D_karman(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Karman', generator)

    def test3D_lid_cavity(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'lid_cavity', generator)

    def test3D_poseuille(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'poiseuille', generator)        