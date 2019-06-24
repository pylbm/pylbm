import importlib
import sys
import os
import pylbm
import pytest
import numpy as np

path = os.path.dirname(__file__) + '/../demo/2D'
path = os.path.abspath(path)


@pytest.fixture
def test2D_case_dir():
    sys.path.append(path)
    yield
    sys.path.pop()

@pytest.mark.slow
@pytest.mark.h5diff(single_reference=True)
@pytest.mark.usefixtures("test2D_case_dir")
@pytest.mark.parametrize('generator', ['numpy', 'cython'])
class Test2D:
    def runtest(self, dx, Tf, name, generator):
        module = importlib.import_module(name)
        return module.run(dx, Tf, generator=generator, with_plot=False)

    def test2D_advection(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'advection', generator)

    def test2D_air_conditioning(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'air_conditioning', generator)

    def test2D_coude(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'bend', generator)

    def test2D_karman_vortex_street(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Karman_vortex_street', generator)

    def test2D_kelvin_Helmoltz(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Kelvin_Helmoltz', generator)

    def test2D_lid_driven_cavity(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'lid_driven_cavity', generator)

    def test2D_orszag_Tang_vortex(self, generator):
        dx, Tf = 2.*np.pi/64, 0.5
        return self.runtest(dx, Tf, 'Orszag_Tang_vortex', generator)

    def test2D_poiseuille(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Poiseuille', generator)

    def test2D_poiseuille_vec(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Poiseuille_vec', generator)

    def test2D_rayleigh_benard(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'Rayleigh-Benard', generator)

    def test2D_shallow_water(self, generator):
        dx, Tf = 1./64, 0.5
        return self.runtest(dx, Tf, 'shallow_water', generator)

