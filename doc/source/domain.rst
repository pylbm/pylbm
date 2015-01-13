Domain
===========

A domain is an object that contains all the geometrical informations
that are needed by the simulation. It is construct as a large box
(segment in 1D, rectangle in 2D, and parallelepipoid in 3D)
with a constant spatial step dx.

A band of phantom cells is added all around the box in order to treat
the boundary conditions.

Once the box being built, geometrical forms can be added or deleted by
using the member's functions. It is useful to consider obstacles.


Class Domain
------------

.. automodule:: pyLBM.domain
   :members:
