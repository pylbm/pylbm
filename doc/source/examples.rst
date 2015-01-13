Examples
========================

Basic examples to start up with pyLBM

The Geometry
------------------

By using the class Geometry, we build the geometrical domain where the
simulation has to be done.

The starting point is a box given by the minimal and maximal values in
each spatial direction.

Basic geometrical forms can then be added or removed: circle,
parallelogram, or triangle.

.. automodule:: pyLBM.Examples_Geometry
   :members:

The Stencil
------------------

The stencil of velocities is built in a very easy way by using the
automatic numbering of velocities.

A stencil object then contains all the informations that are needed by
the Lattice Boltzmann Scheme: in particular the value of each
velocity, the total number of velocities, *etc.*

Note that each elementary scheme owns is proper stencil.

See velocities numbering

        .. image:: /images/Velocities_1D.jpeg

        .. image:: /images/Velocities_2D.jpeg


.. automodule:: pyLBM.Examples_Stencil
   :members:
