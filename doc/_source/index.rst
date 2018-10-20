.. pylbm documentation master file, created by
   sphinx-quickstart on Wed Dec 11 10:32:28 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


pylbm is an all-in-one package for numerical simulations using
Lattice Boltzmann solvers.

pylbm is licensed under the BSD license,
enabling reuse with few restrictions.

Getting started
---------------------------

pylbm can be a simple way to make numerical simulations
by using the Lattice Boltzmann method.

To install pylbm, you have several ways. You can install it using conda ::

    conda install pylbm -c pylbm -c conda-forge

or using the last version on Pypi ::

    pip install pylbm

You can also clone the project ::

    git clone https://github.com/pylbm/pylbm

and then use the command ::

    python setup.py install

or if you don't have root privileges ::

    python setup.py install --user

Once the package is installed
you just have to understand how build a dictionary that will be
understood by pylbm to perform the simulation.
The dictionary should contain all the needed informations as

- the geometry (see :doc:`here<learning_geometry>` for documentation)
- the scheme (see :doc:`here<learning_scheme>` for documentation)
- the boundary conditions (see :doc:`here<learning_bounds>` for documentation)
- another informations like the space step, the scheme velocity, the generator
  of the functions...

To understand how to use pylbm, you have a lot of Python notebooks
in the `tutorial <tutorial.html>`_.

Documentation for users
---------------------------

.. toctree::
   :maxdepth: 2

   The geometry of the simulation <learning_geometry>
   The domain of the simulation <learning_domain>
   The scheme <learning_scheme>
   The boundary conditions <learning_bounds>
   The storage of moments and distribution functions <storage>
   Learning by examples <tutorial>
   Gallery <gallery>
   
Documentation of the code
---------------------------

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
   :maxdepth: 2
   
   stencil <module/module_stencil>
   elements <module/module_elements>
   geometry <module/module_geometry>
   domain <module/module_domain>
   storage <module/module_storage>
   bounds <module/module_bounds>


References
---------------------------

.. [dH92] D. D'HUMIERES, *Generalized Lattice-Boltzmann Equations*,
          Rarefied Gas Dynamics: Theory and Simulations, **159**, pp. 450-458,
          AIAA Progress in astronomics and aeronautics (1992).
.. [D08] F. DUBOIS, *Equivalent partial differential equations of a lattice Boltzmann scheme*,
         Computers and Mathematics with Applications, **55**, pp. 1441-1449 (2008).
.. [G14] B. GRAILLE, *Approximation of mono-dimensional hyperbolic systems: a lattice Boltzmann scheme as a relaxation method*,
         Journal of Comutational Physics, **266** (3179757), pp. 74-88 (2014).
.. [QdHL92] Y.H. QIAN, D. D'HUMIERES, and P. LALLEMAND,
            *Lattice BGK Models for Navier-Stokes Equation*, Europhys. Lett., **17** (6), pp. 479-484 (1992).

Indices and tables
---------------------------

* :ref:`genindex`
* :ref:`search`
