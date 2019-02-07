# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Exact Riemann solvers for the monodimensional hyperbolic systems

  du   d(f(u))
  -- + ------- = 0
  dt     dx

where u is the vector of the unknown and f the convective flux.

The hyperbolic systems can be
 - Burgers
 - shallow water
 - isentropic Euler
 - Euler
"""

from .riemann_solvers import riemann_pb
from .advection_solver import AdvectionSolver
from .burgers_solver import BurgersSolver
from .shallow_water_solver import ShallowWaterSolver
from .p_system_solver import PSystemSolver
from .euler_solver import EulerSolver
from .euler_isothermal_solver import EulerIsothermalSolver
