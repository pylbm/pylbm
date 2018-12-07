"""
p-system
"""

import numpy as np
import matplotlib.pyplot as plt
from .riemann_solvers import GenericSolver, newton


class PSystemSolver(GenericSolver):
    """
        d_t(u1) - d_x(u2) = 0
        d_t(u2) - d_x(p(u1)) = 0

    with p(x) = - x^(-gamma)
    """
    def _read_particular_parameters(self, parameters):
        self.gamma = parameters.get('gamma', 2./3.)
        self.fields = parameters.get('fields name', [r'$u_1$', r'$u_2$'])

    def _compute_p(self, x):
        return -x**(-self.gamma)

    def _compute_dp(self, x):
        return self.gamma*x**(-self.gamma-1)

    def _compute_ddp(self, x):
        return -self.gamma*(self.gamma+1)*x**(-self.gamma-2)

    def _compute_ip(self, x, x_o):
        alpha = -.5*(self.gamma-1)
        return np.sqrt(self.gamma)/alpha*(x**alpha-x_o**alpha)

    def _compute_interstate(self):
        """
        Compute the intermediate state
        """
        x = .5*(self.u_left[0]+self.u_right[0])

        def phi(x):
            return self._f1(x) - self._f2(x)

        u1_star = newton(phi, x, self.epsilon)
        u2_star = self._f1(u1_star)[0]
        self.u_star = np.array([u1_star, u2_star])

    def _rarefaction_1(self, xik):
        u1_k = (-xik/np.sqrt(self.gamma))**(-2/(self.gamma+1))
        u2_k = self._f1(u1_k)[0]
        return np.array([u1_k, u2_k])

    def _rarefaction_2(self, xik):
        u1_k = (xik/np.sqrt(self.gamma))**(-2/(self.gamma+1))
        u2_k = self._f2(u1_k)[0]
        return np.array([u1_k, u2_k])

    def _compute_waves(self):
        """
        Compute the 2 waves, the intermediate state being computed
        """
        self.values.append(self.u_left)
        if self.u_left[0] < self.u_star[0]:  # 1-rarefaction
            self.velocities.append([
                -np.sqrt(self._compute_dp(self.u_left[0])),
                -np.sqrt(self._compute_dp(self.u_star[0]))
            ])
            self.values.append(self._rarefaction_1)
            self.waves.append('rarefaction')
        else:  # 1-shock
            self.velocities.append([
                -(self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0]),
                -(self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_star)
        if self.u_right[0] < self.u_star[0]:  # 2-rarefaction
            self.velocities.append([
                np.sqrt(self._compute_dp(self.u_star[0])),
                np.sqrt(self._compute_dp(self.u_right[0]))
            ])
            self.values.append(self._rarefaction_2)
            self.waves.append('rarefaction')
        else:  # 2-shock
            self.velocities.append([
                -(self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0]),
                -(self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0])
            ])
            self.values.append(None)
            self.waves.append('shock')
        self.values.append(self.u_right)
        return self.velocities

    def _f1(self, u1_star):
        """
        Compute the 1-wave that links the left state
        this wave is parametrized by the first component u1
        """
        u1_left, u2_left = self.u_left[0], self.u_left[1]
        pu1 = self._compute_p(u1_star)
        dpu1 = self._compute_dp(u1_star)
        # ddpu1 = self._compute_ddp(u1_star)
        if u1_star > u1_left:  # 1-rarefaction
            u2_star = u2_left + self._compute_ip(u1_star, u1_left)
            du2_star = np.sqrt(dpu1)
        elif u1_star < u1_left:  # 1-shock
            pu1_left = self._compute_p(u1_left)
            u2_star = u2_left - np.sqrt((pu1-pu1_left)*(u1_star-u1_left))
            du2_star = -(dpu1*(u1_star-u1_left)+pu1-pu1_left) \
                / np.sqrt((pu1-pu1_left)*(u1_star-u1_left))/2
        else:
            u2_star = u2_left
            du2_star = np.sqrt(dpu1)
        return np.array([u2_star, du2_star])

    def _f2(self, u1_star):
        """
        Compute the 2-wave that links the right state
        this wave is parametrized by the first component u1
        """
        u1_right, u2_right = self.u_right[0], self.u_right[1]
        pu1 = self._compute_p(u1_star)
        dpu1 = self._compute_dp(u1_star)
        # ddpu1 = self._compute_ddp(u1_star)
        if u1_star > u1_right:  # 2-rarefaction
            u2_star = u2_right - self._compute_ip(u1_star, u1_right)
            du2_star = -np.sqrt(dpu1)
        elif u1_star < u1_right:  # 2-shock
            pu1_right = self._compute_p(u1_right)
            u2_star = u2_right + np.sqrt((pu1-pu1_right)*(u1_star-u1_right))
            du2_star = (dpu1*(u1_star-u1_right)+pu1-pu1_right) \
                / np.sqrt((pu1-pu1_right)*(u1_star-u1_right))/2
        else:
            u2_star = u2_right
            du2_star = -np.sqrt(dpu1)
        return np.array([u2_star, du2_star])

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
        axes.plot(v_h, vq1, color='orange', label='1-wave', alpha=0.5)
        axes.plot(v_h, vq2, color='navy', label='2-wave', alpha=0.5)
        axes.scatter(self.u_left[0], self.u_left[1],
                     color='orange',
                     s=20,
                     label='left state'
                     )
        axes.scatter(self.u_right[0], self.u_right[1],
                     color='navy',
                     s=20,
                     label='right state'
                     )
        axes.scatter(self.u_star[0], self.u_star[1],
                     color='red',
                     s=20,
                     label='intermediate state'
                     )
        axes.set_title(type(self).__name__)
        axes.set_xlabel(self.fields[0])
        axes.set_ylabel(self.fields[1])
        axes.legend()
