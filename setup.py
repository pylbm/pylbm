from setuptools import setup, Extension
import numpy

setup(
    name           = "pyLBM",
    version        = "0.1",
    description    = "Lattice Boltzmann Method",
    author         = "Benjamin Graille, Loic Gouarin",
    author_email   = "benjamin.graille@math.u-psud.fr, loic.gouarin@math.u-psud.fr",
    packages       = ['pyLBM'],
    package_data   = {'pyLBM': ['../tests/data/domain/*']},
    setup_requires = ['Cython', 'sphinx'],
)
