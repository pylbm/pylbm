"""
isothermal Euler
"""

import numpy as np
import matplotlib.pyplot as plt
from .riemann_solvers import GenericSolver, solve


class EulerIsothermalSolver(GenericSolver):
    """
        d_t(rho)   + d_x(rho u) = 0
        d_t(rho u) + d_x(rho u^2 + c_0^2 rho u) = 0
    
    where c_0 is the speed of sound

    Assuming that p = c_0^2 rho, we have
    E = rho u^2/2 + c_0^2/(gamma-1) rho
    e = c_0^2/(gamma-1) => c_0^2 = (gamma-1) e
    """
    def _read_particular_parameters(self, parameters):
        self.sound_speed = parameters.get('sound_speed', 0.5)
        if self.sound_speed < 0:
            self.sound_speed = -self.sound_speed
        self.fields = parameters.get('fields name', [r'$\rho$', r'$u$'])

    def _compute_interstate(self):
        """
        Compute the intermediate state
        """
        rho_star = solve(
            self._f1, self._f2,
            self.u_left[0], self.u_right[0],
            self.epsilon
        )
        u_star = .5*(self._f1(rho_star)[0]+self._f2(rho_star)[0])
        self.u_star = np.array([rho_star, u_star])

    def _rarefaction_1(self, xik):
        dummy = -(xik - self.velocities[0][0])/self.sound_speed
        rho_k = self.u_left[0] * np.exp(dummy)
        u_k = xik+self.sound_speed
        # u_k = self._f1(rho_k)[0]
        return np.array([rho_k, u_k])

    def _rarefaction_2(self, xik):
        dummy = (xik - self.velocities[1][1])/self.sound_speed
        rho_k = self.u_right[0] * np.exp(dummy)
        u_k = xik-self.sound_speed
        # u_k = self._f2(rho_k)[0]
        return np.array([rho_k, u_k])

    def _compute_waves(self):
        """
        Compute the 2 waves, the intermediate state being computed
        """
        self.values.append(self.u_left)
        if self.u_left[0] > self.u_star[0]:  # 1-rarefaction
            self.velocities.append([
                self.u_left[1] - self.sound_speed,
                self.u_star[1] - self.sound_speed
            ])
            self.values.append(self._rarefaction_1)
            self.waves.append('rarefaction')
        else:  # 1-shock
            self.velocities.append([
                (self.u_star[0]*self.u_star[1]-self.u_left[0]*self.u_left[1]) /
                (self.u_star[0]-self.u_left[0]),
                (self.u_star[0]*self.u_star[1]-self.u_left[0]*self.u_left[1]) /
                (self.u_star[0]-self.u_left[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_star)
        if self.u_right[0] > self.u_star[0]:  # 2-rarefaction
            self.velocities.append([
                self.u_star[1] + self.sound_speed,
                self.u_right[1] + self.sound_speed
            ])
            self.values.append(self._rarefaction_2)
            self.waves.append('rarefaction')
        else:  # 2-shock
            self.velocities.append([
                (self.u_right[0]*self.u_right[1]-self.u_star[0]*self.u_star[1]) /
                (self.u_right[0]-self.u_star[0]),
                (self.u_right[0]*self.u_right[1]-self.u_star[0]*self.u_star[1]) /
                (self.u_right[0]-self.u_star[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_right)

    def _f1(self, rho_star):
        """
        Compute the 1-wave that links the left state
        this wave is parametrized by the first component rho
        """
        rho_left, u_left = self.u_left[0], self.u_left[1]
        if rho_star < rho_left:  # 1-rarefaction
            rapport = np.log(rho_star/rho_left)
            u_star = u_left - self.sound_speed * rapport
            du_star = - self.sound_speed / rho_star
        elif rho_star > rho_left:  # 1-shock
            rapport = np.sqrt(rho_star/rho_left)
            irapport = 1/rapport
            u_star = u_left - self.sound_speed * (
                rapport - irapport
            )
            du_star = - .5*self.sound_speed / rho_star * (
                rapport + irapport
            )
        else:
            u_star = u_left
            du_star = - self.sound_speed / rho_star
        return np.array([u_star, du_star])

    def _f2(self, rho_star):
        """
        Compute the 2-wave that links the right state
        this wave is parametrized by the first component rho
        """
        rho_right, u_right = self.u_right[0], self.u_right[1]
        if rho_star < rho_right:  # 2-rarefaction
            rapport = np.log(rho_star/rho_right)
            u_star = u_right + self.sound_speed * rapport
            du_star = self.sound_speed / rho_star
        elif rho_star > rho_right:  # 2-shock
            rapport = np.sqrt(rho_star/rho_right)
            irapport = 1/rapport
            u_star = u_right + self.sound_speed * (
                rapport - irapport
            )
            du_star = .5*self.sound_speed / rho_star * (
                rapport + irapport
            )
        else:
            u_star = u_right
            du_star = self.sound_speed / rho_star
        return np.array([u_star, du_star])

    def diagram(self):
        hmin = min(self.u_left[0], self.u_right[0])/100
        hmax = max(self.u_left[0], self.u_right[0])*2
        v_h = np.linspace(hmin, hmax, 1025)
        # 1-wave
        vq1 = np.zeros(v_h.shape)
        for i, vhi in enumerate(v_h):
            vq1[i] = self._f1(vhi)[0]
        # 2-wave
        vq2 = np.zeros(v_h.shape)
        for i, vhi in enumerate(v_h):
            vq2[i] = self._f2(vhi)[0]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(vq1, v_h, color='orange', label='1-wave', alpha=0.5)
        axes.plot(vq2, v_h, color='navy', label='2-wave', alpha=0.5)
        axes.scatter(self.u_left[1], self.u_left[0],
                     color='orange',
                     s=20,
                     label='left state'
                     )
        axes.scatter(self.u_right[1], self.u_right[0],
                     color='navy',
                     s=20,
                     label='right state'
                     )
        axes.scatter(self.u_star[1], self.u_star[0],
                     color='red',
                     s=20,
                     label='intermediate state'
                     )
        axes.set_title(type(self).__name__)
        axes.set_xlabel(self.fields[1])
        axes.set_ylabel(self.fields[0])
        axes.legend()
