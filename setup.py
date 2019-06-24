from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"]


MAJOR = "0"
MINOR = "4"
PATCH = "0"
VERSION = "{0}.{1}.{2}".format(MAJOR, MINOR, PATCH)

def write_version_py(filename='pylbm/version.py'):
    a = open(filename, 'w')
    try:
        a.write("version = '{}'".format(VERSION))
    finally:
        a.close()

with open("README.rst", "r") as fh:
    long_description = fh.read()

write_version_py()

setup(
    name           = "pylbm",
    version        = VERSION,
    description    = "A flexible Python package for lattice Boltzmann method",
    long_description = long_description,
    author         = "Benjamin Graille, Loic Gouarin",
    author_email   = "benjamin.graille@math.u-psud.fr, loic.gouarin@polytechnique.edu",
    url            = "https://github.com/pylbm/pylbm",
    license        = "BSD",
    keywords       = "Lattice Boltzmann Methods",
    classifiers    = CLASSIFIERS,
    packages       = find_packages(exclude=['demo', 'doc', 'tests*']),
    package_data   = {'pylbm': ['templates/*']},
    include_package_data=True,
    install_requires=[
                        "numpy",
                        "matplotlib",
                        "sympy >=1.1.1,<1.2",
                        "cython",
                        "h5py",
                        "mpi4py",
                        "colorlog",
                        "colorama",
                        "cerberus",
                        "jinja2",
                      ],
)
