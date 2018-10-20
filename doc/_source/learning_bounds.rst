The Boundary Conditions
******************************

The simulations are performed in a bounded domain with optional obstacles.
Boundary conditions have then to be imposed on all the bounds.
With pylbm, the user can use the classical boundary conditions (classical for the lattice Boltzmann method)
that are already implemented or implement his own conditions.

Note that periodical boundary conditions are used as default conditions.
The corresponding label is :math:`-1`.

For a lattice Boltzmann method, we have to impose the incoming distribution
functions on nodes outside the domain. We describe

   - first, how the bounce back, the anti bounce back, and the Neumann conditions can be used,
   - second, how personal boundary conditions can be implemented.

The classical conditions
========================

The bounce back and anti bounce back conditions
-----------------------------------------------

The bounce back condition (*resp.* anti bounce back) is used to impose
the odd moments (*resp.* even moments) on the bounds.



The Neumann conditions
----------------------


How to implement new conditions
===============================
