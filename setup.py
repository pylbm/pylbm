from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"]


MAJOR = "0"
MINOR = "2"
PATCH = "1"
VERSION = "{0}.{1}.{2}".format(MAJOR, MINOR, PATCH)

def write_version_py(filename='pyLBM/version.py'):
    a = open(filename, 'w')
    try:
        a.write("version = '{}'".format(VERSION))
    finally:
        a.close()

README = open("README.rst").readlines()

write_version_py()

setup(
    name           = "pyLBM",
    version        = VERSION,
    description    = README[0],
    long_description = "".join(README[1:]),
    author         = "Benjamin Graille, Loic Gouarin",
    author_email   = "benjamin.graille@math.u-psud.fr, loic.gouarin@math.u-psud.fr",
    url            = "http://www.math.u-psud.fr/pyLBM",
    license        = "BSD",
    keywords       = "Lattice Boltzmann Methods",
    classifiers    = CLASSIFIERS,
    packages       = find_packages(exclude=['demo', 'doc', 'tests*']),
    #package_data   = {'pyLBM': ['../tests/data/domain/*']},
    include_package_data=True,
    install_requires=[
                      'numpy>=1.9.2',
                      'sympy>=0.7.6',
                      'colorlog>=2.4.0',
                      'Cython>=0.21.1',
                      'mpi4py>=1.3.1',
                      'matplotlib>=1.4.0',
                      'future',
                      'pyevtk>=1.0.0'
                      ],
    dependency_links=[
        "hg+https://bitbucket.org/pauloh/pyevtk"
    ],
    extras_require={'pythran': ["pythran>=0.7.1"],
                    'numba': ["numba>=0.19.1"]
                    },
)
