"""
Shallow water
"""

import numpy as np
import matplotlib.pyplot as plt
from .riemann_solvers import GenericSolver, solve


class ShallowWaterSolver(GenericSolver):
    """
        d_t(h) + d_x(q) = 0
        d_t(q) + d_x(q^2/h+gh^2/2) = 0
    """
    def _read_particular_parameters(self, parameters):
        self.gravity = parameters.get('g', 9.81)
        self.fields = parameters.get('fields name', [r'$h$', r'$u$'])

    def _compute_interstate(self):
        """
        Compute the intermediate state
        """
        h_star = solve(
            self._f1, self._f2,
            self.u_left[0], self.u_right[0],
            self.epsilon
        )
        q_star = .5*(self._f1(h_star)[0]+self._f2(h_star)[0])
        q_star = self._f1(h_star)[0]
        self.u_star = np.array([h_star, q_star])

    def _rarefaction_1(self, xik):
        dummy = (xik-self.velocities[0][0])/3/np.sqrt(self.gravity)
        h_k = (np.sqrt(self.u_left[0]) - dummy)**2
        q_k = self._f1(h_k)[0]
        return np.array([h_k, q_k])

    def _rarefaction_2(self, xik):
        dummy = (xik-self.velocities[1][0])/3/np.sqrt(self.gravity)
        h_k = (np.sqrt(self.u_star[0]) + dummy)**2
        q_k = self._f2(h_k)[0]
        return np.array([h_k, q_k])

    def _compute_waves(self):
        """
        Compute the 2 waves, the intermediate state being computed
        """
        self.values.append(self.u_left)
        if self.u_left[0] > self.u_star[0]:  # 1-rarefaction
            self.velocities.append([
                self.u_left[1]/self.u_left[0] -
                np.sqrt(self.gravity*self.u_left[0]),
                self.u_star[1]/self.u_star[0] -
                np.sqrt(self.gravity*self.u_star[0])
            ])
            self.values.append(self._rarefaction_1)
            self.waves.append('rarefaction')
        else:  # 1-shock
            self.velocities.append([
                (self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0]),
                (self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_star)
        if self.u_right[0] > self.u_star[0]:  # 2-rarefaction
            self.velocities.append([
                self.u_star[1]/self.u_star[0] +
                np.sqrt(self.gravity*self.u_star[0]),
                self.u_right[1]/self.u_right[0] +
                np.sqrt(self.gravity*self.u_right[0])
            ])
            self.values.append(self._rarefaction_2)
            self.waves.append('rarefaction')
        else:  # 2-shock
            self.velocities.append([
                (self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0]),
                (self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_right)

    def _f1(self, h_star):
        """
        Compute the 1-wave that links the left state
        this wave is parametrized by the first component h
        """
        h_left, q_left = self.u_left[0], self.u_left[1]
        u_left, c_left = q_left/h_left, np.sqrt(self.gravity*h_left)
        if h_star < h_left:  # 1-rarefaction
            q_star = h_star * (
                u_left + 2*(c_left - np.sqrt(self.gravity*h_star))
                )
            dq_star = u_left + 2*c_left - 3*np.sqrt(self.gravity*h_star)
        elif h_star > h_left:  # 1-shock
            z = h_star/h_left
            q_star = h_star * (u_left - c_left/np.sqrt(2)*(z-1)*np.sqrt(1+1/z))
            dq_star = u_left - c_left/np.sqrt(2)*(2*z-1)*np.sqrt(1+1/z)\
                + .5*c_left/np.sqrt(2)*(1-1/z)/np.sqrt(1+1/z)
        else:
            q_star = q_left
            dq_star = u_left - np.sqrt(self.gravity*h_star)
        return np.array([q_star, dq_star])

    def _f2(self, h_star):
        """
        Compute the 2-wave that links the right state
        this wave is parametrized by the first component h
        """
        h_right, q_right = self.u_right[0], self.u_right[1]
        u_right, c_right = q_right/h_right, np.sqrt(self.gravity*h_right)
        if h_star < h_right:  # 2-rarefaction
            q_star = h_star * (
                u_right - 2*(c_right - np.sqrt(self.gravity*h_star))
                )
            dq_star = u_right - 2*c_right + 3*np.sqrt(self.gravity*h_star)
        elif h_star > h_right:  # 2-shock
            z = h_star/h_right
            q_star = h_star * (
                u_right + c_right/np.sqrt(2)*(z-1)*np.sqrt(1+1/z)
                )
            dq_star = u_right + c_right/np.sqrt(2)*(2*z-1)*np.sqrt(1+1/z)\
                + .5*c_right/np.sqrt(2)*(1-1/z)/np.sqrt(1+1/z)
        else:
            q_star = q_right
            dq_star = u_right + np.sqrt(self.gravity*h_star)
        return np.array([q_star, dq_star])

    def diagram(self):
        hmin = min(self.u_left[0], self.u_right[0])/100
        hmax = max(self.u_left[0], self.u_right[0])*2
        v_h = np.linspace(hmin, hmax, 1025)
        # 1-wave
        vq1 = np.zeros(v_h.shape)
        for i, vhi in enumerate(v_h):
            vq1[i] = self._f1(vhi)[0]/vhi
        # 2-wave
        vq2 = np.zeros(v_h.shape)
        for i, vhi in enumerate(v_h):
            vq2[i] = self._f2(vhi)[0]/vhi

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(vq1, v_h, color='orange', label='1-wave', alpha=0.5)
        axes.plot(vq2, v_h, color='navy', label='2-wave', alpha=0.5)
        axes.scatter(self.u_left[1]/self.u_left[0], self.u_left[0],
                     color='orange',
                     s=20,
                     label='left state'
                     )
        axes.scatter(self.u_right[1]/self.u_right[0], self.u_right[0],
                     color='navy',
                     s=20,
                     label='right state'
                     )
        axes.scatter(self.u_star[1]/self.u_star[0], self.u_star[0],
                     color='red',
                     s=20,
                     label='intermediate state'
                     )
        axes.set_title(type(self).__name__)
        axes.set_xlabel(self.fields[1])
        axes.set_ylabel(self.fields[0])
        axes.legend()
