.. pylbm documentation master file, created by
   sphinx-quickstart on Wed Dec 11 10:32:28 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pylbm is an all-in-one package for numerical simulations using Lattice Boltzmann solvers.

This package gives all the tools to describe your lattice Boltzmann scheme in 1D, 2D and 3D problems.

We choose the D'Humières formalism to describe the problem. You can have complex geometry with a set of simple shape like circle, sphere, ...

pylbm performs the numerical scheme using Cython, NumPy or Loo.py from the scheme and the domain given by the user. Pythran and Numba wiil be available soon. pylbm has MPI support with mpi4py.

Installation
============

You can install pylbm in several ways

**With mamba or conda**

.. code:: bash

   mamba install pylbm -c conda-forge

.. code:: bash

   conda install pylbm -c conda-forge

**With Pypi**

.. code:: bash

   pip install pylbm

or

.. code:: bash

   pip install pylbm --user

**From source**

You can also clone the project and install the latest version

.. code:: bash

   git clone https://github.com/pylbm/pylbm

To install pylbm from source, we encourage you to create a fresh environment using conda.

.. code:: bash

    conda create -n pylbm_env python

As mentioned at the end of the creation of this environment, you can activate it
using the comamnd line

.. code:: bash

    conda activate pylbm_env

Now, you just have to go into the pylbm directory that you cloned and install
the dependencies

.. code:: bash

    conda install --file requirements-dev.txt -c conda-forge

and then, install pylbm

.. code:: bash

   python setup.py install

Getting started
---------------

pylbm can be a simple way to make numerical simulations
by using the Lattice Boltzmann method.

Once the package is installed
you just have to understand how to build a dictionary that will be
understood by pylbm to perform the simulation.
The dictionary should contain all the needed informations as

- the geometry (see :doc:`here<learning_geometry>` for documentation)
- the scheme (see :doc:`here<learning_scheme>` for documentation)
- another informations like the space step, the scheme velocity, the generator
  of the functions...

.. - the boundary conditions (see :doc:`here<learning_bounds>` for documentation)

To understand how to use pylbm, you have a lot of Python notebooks
in the `tutorial <tutorial.html>`_.

Documentation for users
-----------------------

.. toctree::
   :maxdepth: 1

   The geometry of the simulation <learning_geometry>
   The domain of the simulation <learning_domain>
   The scheme <learning_scheme>
   The scheme analysis  <learning_analysis>
   The storage of moments and distribution functions <storage>
   Learning by examples <tutorial>
   Our Gallery <gallery>

.. The boundary conditions <learning_bounds>

Documentation of the code
-------------------------

.. currentmodule:: pylbm

The most important classes

.. autosummary::
  :toctree: generated/

  Geometry
  Domain
  Scheme
  Simulation

The modules

.. toctree::
   :maxdepth: 1

   stencil <module/module_stencil>
   elements <module/module_elements>
   geometry <module/module_geometry>
   domain <module/module_domain>
   bounds <module/module_bounds>
   algorithms <module/module_algorithm>
   storage <module/module_storage>


References
----------

.. [dH92] D. D'HUMIERES, *Generalized Lattice-Boltzmann Equations*,
          Rarefied Gas Dynamics: Theory and Simulations, **159**, pp. 450-458,
          AIAA Progress in astronomics and aeronautics (1992).
.. [D08] F. DUBOIS, *Equivalent partial differential equations of a lattice Boltzmann scheme*,
         Computers and Mathematics with Applications, **55**, pp. 1441-1449 (2008).
.. [G14] B. GRAILLE, *Approximation of mono-dimensional hyperbolic systems: a lattice Boltzmann scheme as a relaxation method*,
         Journal of Computational Physics, **266** (3179757), pp. 74-88 (2014).
.. [QdHL92] Y.H. QIAN, D. D'HUMIERES, and P. LALLEMAND,
            *Lattice BGK Models for Navier-Stokes Equation*, Europhys. Lett., **17** (6), pp. 479-484 (1992).

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
