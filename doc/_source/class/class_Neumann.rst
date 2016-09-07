======================================
the classes for Neumann
======================================

Four classes are given for Neumann conditions:
the principle is just to copy the value of the distribution functions
on the nodes that are just outside the domain. The copy is done along

   - the direction of the velocity for the Neumann condition,
   - the x-direction for the Neumann_x condition,
   - the y-direction for the Neumann_y condition,
   - the z-direction for the Neumann_z condition.

the class Neumann
-------------------------

.. autoclass:: pyLBM.boundary.Neumann
   :members:
   :no-undoc-members:


the class Neumann_x
-------------------------

.. autoclass:: pyLBM.boundary.Neumann_x
   :members:
   :no-undoc-members:


the class Neumann_y
-------------------------

.. autoclass:: pyLBM.boundary.Neumann_y
   :members:
   :no-undoc-members:


the class Neumann_z
-------------------------

.. autoclass:: pyLBM.boundary.Neumann_z
   :members:
   :no-undoc-members:
