pyLBM
=====

|build status| |Doc badge| |Gitter Badge|

.. |Build Status| image:: https://travis-ci.org/pylbm/pylbm.svg?branch=develop
   :target: https://travis-ci.org/pylbm/pylbm
.. |Gitter Badge| image:: https://badges.gitter.im/pylbm/pylbm.svg
   :alt: Join the chat at https://gitter.im/pylbm/pylbm
   :target: https://gitter.im/pylbm/pylbm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Doc badge| image:: https://readthedocs.org/projects/pylbm/badge/?version=develop
   :target: http://pylbm.readthedocs.io/en/develop/
   
pyLBM is an all-in-one package for numerical simulations using Lattice Boltzmann solvers.

This package gives all the tools to describe your lattice Boltzmann scheme in 1D, 2D and 3D problems.

We choose the D'Humières formalism to describe the problem. You can have complex geometry with a set of simple shape like circle, sphere, ...

pyLBM performs the numerical scheme using Cython, Pythran or Numba from the scheme and the domain given by the user. pyLBM has MPI support with mpi4py.

Installation
============

You can install the last version on Pypi

  pip install pyLBM

You can also clone the project

  git clone https://github.com/pylbm/pylbm

and then use the command

  pip install -r requirements.txt
  
  python setup.py install

Getting started
================

To understand how to use pyLBM, you have a lot of Python notebooks on our website

`<http://www.math.u-psud.fr/pyLBM/tutorial.html>`_

For more information, take a look at the documentation

`<http://www.math.u-psud.fr/pyLBM>`_

