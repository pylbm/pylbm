from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name           = "pyLBM",
    version        = "0.1",
    description    = "Lattice Boltzmann Method",
    author         = "Benjamin Graille, Loic Gouarin",
    author_email   = "benjamin.graille@math.u-psud.fr, loic.gouarin@math.u-psud.fr",
    packages       = ['pyLBM'],
    package_data   = {'pyLBM': ['../tests/data/domain/*']},
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pyLBM.bc_utils", ["pyLBM/bc_utils.pyx"],)],
    setup_requires = ['Cython', 'sphinx'],
)
