import importlib
import sys
import os
import pylbm
import pytest
import numpy as np

path = os.path.dirname(__file__) + '/../demo/2D'
path = os.path.abspath(path)


@pytest.fixture
def test_case_dir():
    sys.path.append(path)
    yield
    sys.path.pop()

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

    def test_air_conditioning(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'air_conditioning', generator)

    def test_coude(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'coude', generator)

    def test_karman_vortex_street(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Karman_vortex_street', generator)

    def test_kelvin_Helmotz(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Kelvin_Helmotz', generator)

    def test_lid_driven_cavity(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'lid_driven_cavity', generator)

    def test_orszag_Tang_vortex(self, generator):
        dx, Tf = 2.*np.pi/64, 0.5
        return self.runtest(dx, Tf, 'Orszag_Tang_vortex', generator)

    def test_poiseuille(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Poiseuille', generator)

    def test_poiseuille_vec(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Poiseuille_vec', generator)

    def test_rayleigh_benard(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Rayleigh-Benard', generator)

    def test_shallow_water(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'shallow_water', generator)

