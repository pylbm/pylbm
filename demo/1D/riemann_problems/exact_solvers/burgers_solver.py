"""
Burgers
"""

import numpy as np
from .riemann_solvers import GenericSolver


class BurgersSolver(GenericSolver):
    """
        d_t(u) + d_x(u^2/2) = 0
    """
    def _read_particular_parameters(self, parameters):
        self.fields = parameters.get('fields name', [r'$u$'])

    def _rarefaction(self, xik):
        return np.array([xik])

    def _compute_waves(self):
        """
        Compute the wave (rarefaction or shock)
        """
        self.values.append(self.u_left)
        if self.u_left[0] < self.u_right[0]:  # rarefaction
            self.velocities.append([
                self.u_left[0], self.u_right[0]
            ])
            self.values.append(self._rarefaction)
            self.waves.append('rarefaction')
        else:  # shock
            self.velocities.append([
                .5*(self.u_left[0]+self.u_right[0]),
                .5*(self.u_left[0]+self.u_right[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_right)
        return self.velocities
