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


def riemann_pb(x, xmid, u_left, u_right):
    """
    initial condition with a Riemann problem

    Parameters
    ----------

    x : ndarray
        spatial mesh

    xmid : double
        position of the discontinuity

    u_left : double
        left value of the field

    u_right : double
        right value of the field

    Returns
    -------

    vect_u
        ndarray
    """
    vect_u = np.empty(x.shape)
    vect_u[x < xmid] = u_left
    vect_u[x >= xmid] = u_right
    return vect_u


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


def solve(f1, f2, xa, xb, eps, nitermax=1000):
    """
    solve f1(x) = f2(x) with a Newton method
    the first step is the computation of the Newton initialization
    method proposed by Francois Dubois
    
    # intersection of the two tangents
    #    (y-ya) - df1(xa)(x-xa) = 0
    #    (y-yb) - df2(xb)(x-xb) = 0
    #  with ya = f1(xa) and yb = f2(xb)
    #  =>    y = ya + df1(xa)(x-xa) = yb + df2(xb)(x-xb)
    #  =>    (df2(xb) - df1(xa)) x =  ya - xa df1(xa) - yb + xb df2(xb)
    
    problem for two rarefactions waves: the intersection point is negative
    in this case switch xa and xb...
    """
    ya, dya = f1(xa)
    yb, dyb = f2(xb)
    x = (ya - xa * dya - yb + xb * dyb) / (dyb - dya)
    if x <= 0:
        ya, dya = f2(xa)
        yb, dyb = f1(xb)
        x = (ya - xa * dya - yb + xb * dyb) / (dyb - dya)
    if x <= 0:
        print("Problem to compute interstate !!!")
    
    def phi(x):
        return f1(x) - f2(x)

    return newton(phi, x, eps, nitermax)



class GenericSolver(object):
    """
    generic class for the Riemann solver
    """
    def __init__(self, parameters):
        self._read_parameters(parameters)
        self._read_particular_parameters(parameters)
        self.n = self.u_left.size
        self.velocities = []
        self.values = []
        self.waves = []
        self._compute_interstate()
        self._compute_waves()

    def _read_parameters(self, parameters):
        self.name = parameters.get('name', None)
        self.pos_disc = parameters.get("jump abscissa", 0)
        self.u_left = parameters.get('left state', None)
        self.u_right = parameters.get('right state', None)
        self.epsilon = parameters.get('Newton precision', 1.e-10)
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
        y = np.zeros((self.n, x.size))
        if t == 0:
            for k in range(self.n):
                y[k, x < self.pos_disc] = self.u_left[k]
                y[k, x >= self.pos_disc] = self.u_right[k]
        else:
            vxi = (x-self.pos_disc) / t
            for k in range(self.n):
                y[k, :] = self.values[0][k]  # left value
            for k, xik in enumerate(vxi):
                for i in range(len(self.velocities)):
                    if self.waves[i] == 'rarefaction' \
                       and xik > self.velocities[i][0] \
                       and xik < self.velocities[i][1]:
                        y[:, k] = self.values[2*i+1](xik)
                    if xik >= self.velocities[i][1]:
                        y[:, k] = self.values[2*i+2]
        return y

    def diagram(self):
        """
        plot the diagram of the waves
        """
        print("No diagram to plot !!!")
