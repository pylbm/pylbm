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

import numpy as np
import matplotlib.pyplot as plt


def newton(f, x, eps, nitermax=1000):
    """
    Newton method to solve f(x)=0
    the algorithm is stopped when |f(x)| < eps
    """
    xold = x+2*eps
    f_x, df_x = f(x)
    niter = 0
    while abs(f_x) > eps and abs(x-xold) > eps and niter < nitermax:
        xold = x
        x -= f_x/df_x
        f_x, df_x = f(x)
        niter += 1
    if niter == nitermax:
        print("Newton has not converged: error = {:10.3e}".format(f_x))
    return x


class GenericSolver(object):
    """
    generic class for the Riemann solver
    """
    def __init__(self, parameters):
        self._read_parameters(parameters)
        self._read_particular_parameters(parameters)
        self._compute_interstate()
        self._compute_waves()
        self.n = self.u_left.size

    def _read_parameters(self, parameters):
        self.name = parameters.get('name', None)
        self.pos_disc = parameters.get("jump abscissa", 0)
        self.u_left = parameters.get('left state', None)
        self.u_right = parameters.get('right state', None)
        self.epsilon = parameters.get('Newton precision', 1.e-15)
        if self.u_left is None or self.u_right is None:
            print("*"*80)
            print("Error in the parameters of {}".format(type(self).__name__))
            print("You should define 'left state' and 'right state'")
            print("*"*80)
        else:
            self.u_left = np.array(self.u_left)
            self.u_right = np.array(self.u_right)
        self.syst_size = self.u_left.size
        self.fields = [None]*self.syst_size

    def _read_particular_parameters(self, parameters):
        pass

    def _compute_interstate(self):
        self.u_star = np.nan * np.ones(self.u_left.shape)

    def _compute_waves(self):
        return [None]*4

    def plot(self, x, t):
        """
        plot the exact solution at time t on the mesh x
        """
        y = self.evaluate(x, t)
        fig = plt.figure()
        for k in range(self.syst_size):
            axes = fig.add_subplot(self.syst_size, 1, k+1)
            axes.plot(x, y[k], color='navy', alpha=0.5)
            axes.set_ylabel(self.fields[k])
            if k == 0:
                name = type(self).__name__
                if self.name is not None:
                    name += " (" + self.name + ")"
                axes.set_title(name)
            if k < self.syst_size-1:
                axes.get_xaxis().set_visible(False)

    def evaluate(self, x, t):  # pylint: disable=unused-argument
        """
        evaluate the exact solution at time t on the mesh x
        """
        return np.nan * np.ones(x.shape)

    def diagram(self):
        """
        plot the diagram of the waves
        """


class AdvectionSolver(GenericSolver):
    """
        d_t(u) + c d_x(u) = 0
    """
    def _read_particular_parameters(self, parameters):
        self.velocity = parameters.get('velocity', 1)
        self.fields = parameters.get('fields name', [r'$u$'])

    def evaluate(self, x, t):
        y = np.zeros(x.shape)
        y[x < self.pos_disc+self.velocity*t] = self.u_left
        y[x >= self.pos_disc+self.velocity*t] = self.u_right
        return y


class BurgersSolver(GenericSolver):
    """
        d_t(u) + d_x(u^2/2) = 0
    """
    def _read_particular_parameters(self, parameters):
        self.fields = parameters.get('fields name', [r'$u$'])

    def _compute_waves(self):
        """
        Compute the wave (rarefaction or shock)
        """
        self.velocities = []
        if self.u_left[0] < self.u_right[0]:  # rarefaction
            self.velocities.append([
                self.u_left[0], self.u_right[0]
            ])
        else:  # shock
            self.velocities.append([
                .5*(self.u_left[0]+self.u_right[0]),
                .5*(self.u_left[0]+self.u_right[0])
            ])
        return self.velocities

    def evaluate(self, x, t):
        y = np.zeros(x.shape)

        if t == 0:
            y[x < self.pos_disc] = self.u_left
            y[x >= self.pos_disc] = self.u_right
        else:
            vxi = (x - self.pos_disc)/t
            for k, xik in enumerate(vxi):
                if xik < self.velocities[0][0]:
                    y[k] = self.u_left
                elif xik < self.velocities[0][1]:
                    y[k] = xik
                else:
                    y[k] = self.u_right
        return y


class ShallowWaterSolver(GenericSolver):
    """
        d_t(h) + d_x(q) = 0
        d_t(q) + d_x(q^2/h+gh^2/2) = 0
    """
    def _read_particular_parameters(self, parameters):
        self.gravity = parameters.get('g', 9.81)
        self.fields = parameters.get('fields name', [r'$h$', r'$q$'])

    def _compute_interstate(self):
        """
        Compute the intermediate state
        """
        x = max(self.u_left[0], self.u_right[0])

        def phi(x):
            return self._f1(x) - self._f2(x)

        h_star = newton(phi, x, self.epsilon)
        q_star = self._f1(h_star)[0]
        self.u_star = np.array([h_star, q_star])

    def _compute_waves(self):
        """
        Compute the 2 waves, the intermediate state being computed
        """
        self.velocities = []
        if self.u_left[0] > self.u_star[0]:  # 1-rarefaction
            self.velocities.append([
                self.u_left[1]/self.u_left[0] -
                np.sqrt(self.gravity*self.u_left[0]),
                self.u_star[1]/self.u_star[0] -
                np.sqrt(self.gravity*self.u_star[0])
            ])
        else:  # 1-shock
            self.velocities.append([
                (self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0]),
                (self.u_star[1]-self.u_left[1]) /
                (self.u_star[0]-self.u_left[0])
            ])
        if self.u_right[0] > self.u_star[0]:  # 2-rarefaction
            self.velocities.append([
                self.u_star[1]/self.u_star[0] +
                np.sqrt(self.gravity*self.u_star[0]),
                self.u_right[1]/self.u_right[0] +
                np.sqrt(self.gravity*self.u_right[0])
            ])
        else:  # 2-shock
            self.velocities.append([
                (self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0]),
                (self.u_right[1]-self.u_star[1]) /
                (self.u_right[0]-self.u_star[0])
            ])
        return self.velocities

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

    def evaluate(self, x, t):
        y = np.zeros((self.n, x.size))
        if t == 0:
            for k in range(self.n):
                y[k, x < self.pos_disc] = self.u_left[k]
                y[k, x >= self.pos_disc] = self.u_right[k]
        else:
            vxi = (x-self.pos_disc) / t
            for k, xik in enumerate(vxi):
                if xik < self.velocities[0][0]:
                    y[0, k] = self.u_left[0]
                    y[1, k] = self.u_left[1]
                elif xik < self.velocities[0][1]:
                    dummy = (xik-self.velocities[0][0])/3/np.sqrt(self.gravity)
                    y[0, k] = (np.sqrt(self.u_left[0]) - dummy)**2
                    y[1, k] = self._f1(y[0, k])[0]
                elif xik < self.velocities[1][0]:
                    y[0, k] = self.u_star[0]
                    y[1, k] = self.u_star[1]
                elif xik < self.velocities[1][1]:
                    dummy = (xik-self.velocities[1][0])/3/np.sqrt(self.gravity)
                    y[0, k] = (np.sqrt(self.u_star[0]) + dummy)**2
                    y[1, k] = self._f2(y[0, k])[0]
                else:
                    y[0, k] = self.u_right[0]
                    y[1, k] = self.u_right[1]
        return y

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
        axes.set_title(type(self).__name__ + " (" + self.name + ")")
        axes.set_xlabel(r'$h$')
        axes.set_ylabel(r'$q$')
        axes.legend()


def test_cases_riemann(num=None):
    """
    test the Riemann solver
    """
    if num is None:
        num = list(range(4))
    if isinstance(num, int):
        num = [num]
    dico = {
        0: {
            'name': 'shock-shock',
            'left state': [1, 2],
            'right state': [2, 1],
        },
        1: {
            'name': 'shock-rarefaction',
            'left state': [2, 1],
            'right state': [4, -1],
        },
        2: {
            'name': 'rarefaction-shock',
            'left state': [4, 2],
            'right state': [2, -1],
        },
        3: {
            'name': 'rarefaction-rarefaction',
            'left state': [3, 1],
            'right state': [3, 4],
        },
    }
    for n in num:
        solver_cfg = dico.get(n, None)
        if solver_cfg is not None:
            solver_cfg['Newton precision'] = 1.e-15
            exact_solution = ShallowWaterSolver(solver_cfg)
            exact_solution.diagram()
            x = np.linspace(-1, 1, 1025)
            t = 0.1
            exact_solution.plot(x, t)
    plt.show()

if __name__ == "__main__":
    test_cases_riemann(2)
