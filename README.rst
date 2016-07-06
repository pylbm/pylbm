|build status|

pyLBM is an all-in-one package for numerical simulations using Lattice Boltzmann solvers.

This package gives all the tools to describe your lattice Boltzmann scheme in 1D, 2D and 3D problems.

We choose the D'Humières formalism to describe the problem. You can have complex geometry with a set of simple shape like circle, sphere, ...

pyLBM performs the numerical scheme using Cython, Pythran or Numba from the scheme and the domain given by the user. pyLBM has MPI support with mpi4py.

Installation
============

To install pyLBM, you have several ways. You can clone the project

  git clone https://gitlab.com/gouarin/pylbm

and then use the command

  python setup.py install

You can also install the last version on Pypi

  pip install pyLBM

Getting started
================

To understand how to use pyLBM, you have a lot of Python notebooks on our website

`<http://www.math.u-psud.fr/pyLBM/tutorial.html>`_

For more information, take a look at the documentation

`<http://www.math.u-psud.fr/pyLBM>`_

.. |Build Status| image:: https://travis-ci.org/pylbm/pylbm.svg?branch=develop
-   :target: https://travis-ci.org/pylbm/pylbm
