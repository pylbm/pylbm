.. pyLBM documentation master file, created by
   sphinx-quickstart on Wed Dec 11 10:32:28 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


pyLBM is an all-in-one package for numerical simulations using
Lattice Boltzmann solvers.

pyLBM is licensed under the BSD license,
enabling reuse with few restrictions.

First steps
---------------------------

pyLBM can be a simple way to make numerical simulations
by using the Lattice Boltzmann method.

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

Documentation for users
---------------------------

.. toctree::
   :maxdepth: 2

   The geometry of the simulation <learning_geometry>
   The scheme <learning_scheme>
   The boundary conditions <learning_bounds>


Documentation of the code
---------------------------

.. toctree::
  :maxdepth: 2

  The class Geometry <class_geometry>
  The module elements <module_elements>
  The module stencil <module_stencil>
  The class Domain <class_domain>
  The class Scheme <class_scheme>
  The module generator <module_generator>
  The class Simulation <class_simulation>

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
