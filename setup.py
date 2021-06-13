from setuptools import setup, find_packages

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Software Development",
    'Topic :: Software Development :: Code Generators',
    "Topic :: Scientific/Engineering",
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"]


MAJOR = "0"
MINOR = "7"
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
    name             = "pylbm",
    version          = VERSION,
    description      = "A flexible Python package for lattice Boltzmann method",
    long_description = long_description,
    author           = "Benjamin Graille, Loic Gouarin",
    author_email     = "benjamin.graille@universite-paris-saclay.fr, loic.gouarin@polytechnique.edu",
    url              = "https://github.com/pylbm/pylbm",
    license          = "BSD",
    keywords         = "Lattice Boltzmann Methods",
    classifiers      = CLASSIFIERS,
    packages         = find_packages(exclude=['demo', 'doc', 'tests*']),
    package_data     = {'pylbm': ['templates/*']},
    python_requires  = '>=3.6',
    install_requires = [
                        "numpy",
                        "numpy-stl",
                        "matplotlib",
                        "sympy>=1.3",
                        "cython",
                        "h5py",
                        "mpi4py",
                        "colorlog",
                        "colorama",
                        "cerberus",
                        "jinja2<3.0.0",
                      ],
    extras_require  = {
        'gpu': ['pyopencl', 'loo.py==2017.2']
    }
)
