pylbm
=====

|Binder| |Travis| |Appveyor| |Doc badge| |Gitter Badge|

.. |Binder| image:: https://mybinder.org/badge.svg 
   :target: https://mybinder.org/v2/gh/pylbm/pylbm/develop
.. |Travis| image:: https://travis-ci.org/pylbm/pylbm.svg?branch=develop
   :target: https://travis-ci.org/pylbm/pylbm
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/lm3gufe7njj29s0t/branch/develop?svg=true
   :target: https://ci.appveyor.com/project/pylbm/pylbm
.. |Gitter Badge| image:: https://badges.gitter.im/pylbm/pylbm.svg
   :alt: Join the chat at https://gitter.im/pylbm/pylbm
   :target: https://gitter.im/pylbm/pylbm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Doc badge| image:: https://readthedocs.org/projects/pylbm/badge/?version=develop
   :target: http://pylbm.readthedocs.io/en/develop/
   
pylbm is an all-in-one package for numerical simulations using Lattice Boltzmann solvers.

This package gives all the tools to describe your lattice Boltzmann scheme in 1D, 2D and 3D problems.

We choose the D'Humières formalism to describe the problem. You can have complex geometry with a set of simple shape like circle, sphere, ...

pylbm performs the numerical scheme using Cython, NumPy or Loo.py from the scheme and the domain given by the user. Pythran and Numba wiil be available soon. pylbm has MPI support with mpi4py.

Installation
============

You can install pylbm in several ways

With conda
----------

.. code::

   conda install pylbm -c pylbm -c conda-forge
  
With Pypi
---------

.. code::

   pip install pylbm

or
  
.. code::

   pip install pylbm --user

From source
-----------

You can also clone the project

.. code::

   git clone https://github.com/pylbm/pylbm

and then use the command

.. code::

   python setup.py install

For more information, take a look at the documentation

`<http://pylbm.readthedocs.io>`_

