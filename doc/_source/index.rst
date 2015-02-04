.. pyLBM documentation master file, created by
   sphinx-quickstart on Wed Dec 11 10:32:28 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


pyLBM is an all-in-one package for numerical simulations using
Lattice Boltzmann solvers.
It is written in python 2.7 for the moment.

pyLBM is licensed under the BSD license,
enabling reuse with few restrictions.

First steps
---------------------------

pyLBM can be a simple way to make numerical simulations
by using the Lattice Boltzmann method.
The module can be downloaded `here <http://www.math.u-psud.fr/~pyLBM/>`_.
Before installing, please check the required and optional modules (see below).

Once the module is installed by the command::

    python setup.py install

you just have to understand how build a dictionary that will be
understood by pyLBM to perform the simulation.
The dictionary should contain all the needed informations as

- the geometry (see :doc:`here<learning_geometry>` for documentation)
- the scheme (see :doc:`here<learning_scheme>` for documentation)
- the boundary conditions (see :doc:`here<learning_bounds>` for documentation)
- another informations like the space step, the scheme velocity, the generator
  of the functions...

Documentation for users:
---------------------------

.. toctree::
   :maxdepth: 2

   The geometry of the simulation <learning_geometry>
   The scheme <learning_scheme>
   The boundary conditions <learning_bounds>


Documentation of the code:
---------------------------

.. toctree::
  :maxdepth: 2

  The class Geometry <class_geometry>
  The module elements <module_elements>
  The module stencil <module_stencil>
  The class Domain <class_domain>
  The class Scheme <class_scheme>
  The class Simulation <class_simulation>

Requires:
---------------------------

   * numpy
   * sympy
   * matplotlib
   * time

Optionals:
---------------------------

   * cython
   * mpi4py
   * vtk
   * vispy

Indices and tables
---------------------------

* :ref:`genindex`
* :ref:`search`
