The Geometry of the simulation
=================================

With pyLBM, the numerical simulations can be performed in a domain
with a complex geometry. This geometry is construct without considering a
particular mesh but only with geometrical objects.

Fist, the domain is put into a box: a segment in 1D, a rectangle in 2D, and
a rectangular parallelepipoid in 3D.

Then, the domain is modified by adding or deleting some elementary shapes.
In 2D, the elementary shapes are

* a circle (:py:class:`pyLBM.Circle`)
* a parallelogram (:py:class:`pyLBM.Parallelogram`)
* a triangle (:py:class:`pyLBM.Triangle`)

In 3D, the elementary shapes are not yet implemented.

The class Geometry
------------------------------

.. autoclass:: pyLBM.Geometry
   :members:
   :private-members:
   :special-members:

The classes of elements
------------------------------

.. automodule:: pyLBM.elements
   :members:
