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
.. list-table::
   :header-rows: 1

   * - class
     - :math:`\alpha(\delta)`
     - post-collision correction
   * - :py:class:`BouzidiBounceBack <pylbm.boundary.BouzidiBounceBack>`
     - :math:`2` or :math:`1/\delta`
     - no
   * - :py:class:`CLIBounceBack <pylbm.boundary.CLIBounceBack>`
     - :math:`4/(1+2\delta)`
     - not needed
   * - :py:class:`YLIBounceBack <pylbm.boundary.YLIBounceBack>`
     - :math:`2/(1+\delta)`
     - no
   * - :py:class:`ParYLIBounceBack <pylbm.boundary.ParYLIBounceBack>`
     - :math:`2/(1+\delta)`
     - yes
   * - :py:class:`ParBFLBounceBack <pylbm.boundary.ParBFLBounceBack>`
     - :math:`2` or :math:`1/\delta`
     - yes

The anti bounce back has the same family with the symmetric parts:
:py:class:`BouzidiAntiBounceBack <pylbm.boundary.BouzidiAntiBounceBack>`,
:py:class:`LMKCAntiBounceBack <pylbm.boundary.LMKCAntiBounceBack>` and
:py:class:`ParLMKAAntiBounceBack <pylbm.boundary.ParLMKAAntiBounceBack>`
(see :py:class:`LinearAntiBounceBack <pylbm.boundary.LinearAntiBounceBack>`).

The post-collision correction adds the change of the populations during the
previous time step, collision and source included. It makes the scheme
parametrized: with a TRT collision, the steady solution only depends on the
product :math:`\Lambda^+\Lambda^-` of the relaxation parameters, as for the
bounce back. For example, a force-driven Poiseuille flow is exact for
:math:`\Lambda^+\Lambda^- = \delta^2/2` with a wall momentum
:math:`\Lambda^- F`, whatever the viscosity (the momentum of the stored
populations is then the velocity plus :math:`F/2`).
:py:class:`CLIBounceBack <pylbm.boundary.CLIBounceBack>` and
:py:class:`LMKCAntiBounceBack <pylbm.boundary.LMKCAntiBounceBack>` are
parametrized without correction.

The correction reads the populations of the previous time step. With the
NumPy generator, the simulation then stores them in a second array, which
doubles the memory of the distribution functions. If the populations are
modified between two time steps (``sol.F_halo[...] = ...``), the correction
of the next time step is wrong.

To use another interpolation, subclass ``LinearBounceBack`` or
``LinearAntiBounceBack``, define the method ``alpha`` and set
``post_collision_correction``.



The Neumann conditions
----------------------


How to implement new conditions
===============================
