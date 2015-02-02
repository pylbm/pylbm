# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys
import cmath
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from .domain import Domain
from .scheme import Scheme
from .geometry import Geometry
from .stencil import Stencil
from .boundary import Boundary

from pyLBM import utils

from .logs import __setLogger
log = __setLogger(__name__)

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

class Simulation:
    """
    create a class simulation

    Parameters
    ----------

    domain : object of class :py:class:`pyLBM.Domain`
    scheme : object of class :py:class:`pyLBM.Scheme`
    type :   optional argument (default value is 'float64')

    Attributs
    ---------

    dim :        spatial dimension
    type :       the type of the values
    domain :     the domain given in argument
    scheme :     the scheme given in argument
    _m :         a numpy array that contains the values of the moments in each point
    _F :         a numpy array that contains the values of the distribution functions in each point

    Methods
    -------

    initialization :     initialize all the array
    transport :          compute the transport phase (modifies the array _F)
    relaxation :         compute the relaxation phase (modifies the array _m)
    equilibrium :        compute the equilibrium
    f2m :                compute the moments _m from the distribution _F
    m2f :                compute the distribution _F from the moments _m
    boundary_condition : compute the boundary conditions (modifies the array _F)
    one_time_step :      compute a complet time step combining
      boundary_condition, transport, f2m, relaxation, m2f

    """
    def __init__(self, dico, domain=None, scheme=None, type='float64'):
        self.type = type
        self.order = 'C'

        log.info('Build the domain')
        try:
            if domain is not None:
                self.domain = domain
            else:
                self.domain = Domain(dico)
        except KeyError:
            log.error('Error in the creation of the domain: wrong dictionnary')
            sys.exit()

        log.info('Build the scheme')
        try:
            if scheme is not None:
                self.scheme = scheme
            else:
                self.scheme = Scheme(dico)
        except KeyError:
            log.error('Error in the creation of the scheme: wrong dictionnary')
            sys.exit()

        self.t = 0.
        self.nt = 0
        self.dt = self.domain.dx / self.scheme.la
        try:
            assert self.domain.dim == self.scheme.dim
        except:
            log.error('Solution: the dimension of the domain and of the scheme are not the same\n')
            sys.exit()

        self.dim = self.domain.dim


        log.info('Build arrays')
        #self.nv_on_beg = nv_on_beg
        self.nv_on_beg = self.scheme.nv_on_beg

        if self.nv_on_beg:
            msize = [self.scheme.stencil.nv_ptr[-1]] + self.domain.Na[::-1]
            self._m = np.empty(msize, dtype=self.type, order=self.order)
            self._F = np.empty(msize, dtype=self.type, order=self.order)
        else:
            msize = self.domain.Na[::-1] + [self.scheme.stencil.nv_ptr[-1]]
            self._m = np.empty(msize, dtype=self.type, order=self.order)
            self._F = np.empty(msize, dtype=self.type, order=self.order)
            self._Fold = np.empty(msize, dtype=self.type, order=self.order)

        # self.m = [np.empty([self.scheme.stencil.nv[k]] + self.domain.Na, dtype=self.type, order=self.order) for k in range(self.scheme.nscheme)]
        # self.F = [np.empty([self.scheme.stencil.nv[k]] + self.domain.Na, dtype=self.type, order=self.order) for k in range(self.scheme.nscheme)]

        log.info('Build boundary conditions')
        self.bc = Boundary(self.domain, dico)

        log.info('Initialization')
        self.initialization(dico)

        #computational time measurement
        self.cpu_time = {'relaxation':0.,
                         'transport':0.,
                         'f2m_m2f':0.,
                         'boundary_conditions':0.,
                         'total':0.,
                         'number_of_iterations':0,
                         'MLUPS':0.,
                         }

    @utils.item2property
    def m(self, i, j):
        if type(j) is slice:
            jstart, jstop = j.start, j.stop
            if j.start is None:
                jstart = 0
            if j.stop is None:
                jstop = self.scheme.stencil.nv[i] - 1
            jj = slice(self.scheme.stencil.nv_ptr[i] + jstart,
                       self.scheme.stencil.nv_ptr[i] + jstop)
            if self.nv_on_beg:
                return self._m[jj]
            else:
                return self._m[:, :, jj]
        if self.nv_on_beg:
            return self._m[self.scheme.stencil.nv_ptr[i] + j]
        else:
            return self._m[:, :, self.scheme.stencil.nv_ptr[i] + j]

    @m.setter
    def m(self, i, j, value):
        if self.nv_on_beg:
            self._m[self.scheme.stencil.nv_ptr[i] + j] = value
        else:
            self._m[:, :, self.scheme.stencil.nv_ptr[i] + j] = value

    @utils.item2property
    def F(self, i, j):
        if self.nv_on_beg:
            return self._F[self.scheme.stencil.nv_ptr[i] + j]
        else:
            return self._F[:, :, self.scheme.stencil.nv_ptr[i] + j]

    @F.setter
    def F(self, i, j, value):
        if self.nv_on_beg:
            self._F[self.scheme.stencil.nv_ptr[i] + j] = value
        else:
            self._F[:, :, self.scheme.stencil.nv_ptr[i] + j] = value

    def __str__(self):
        s = "Simulation informations\n"
        s += self.domain.__str__()
        s += self.scheme.__str__()
        return s

    def initialization(self, dico):
        # type of initialization
        # by default, the initialization is on the moments
        # else, it could be distributions
        inittype = dico.get('inittype', 'moments')
        if self.dim == 1:
            x = self.domain.x[0]
            coords = (x,)
        elif self.dim == 2:
            #x = self.domain.x[0][:,np.newaxis]
            #y = self.domain.x[1][np.newaxis, :]
            x = self.domain.x[0][np.newaxis, :]
            y = self.domain.x[1][: ,np.newaxis]
            coords = (x, y)

        schemes = dico['schemes']
        for ns, s in enumerate(schemes):
            for k, v in s['init'].iteritems():
                f = v[0]
                extraargs = v[1] if len(v) == 2 else ()
                fargs = coords + extraargs
                if inittype == 'moments':
                    if self.nv_on_beg:
                        self._m[self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                    else:
                        log.debug('tricky for the treatment of the dimension')
                        if self.dim == 1:
                            self._m[:, self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                        elif self.dim == 2:
                            self._m[:, :, self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                        elif self.dim == 3:
                            self._m[:, :, :, self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                        else:
                            log.error('Problem of dimension in initialization')
                elif inittype == 'distributions':
                    if self.nv_on_beg:
                        self._F[self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                    else:
                        self._F[:, :, self.scheme.stencil.nv_ptr[ns] + k] = f(*fargs)
                else:
                    sss = 'Error in the creation of the scheme: wrong dictionnary\n'
                    sss += 'the key `inittype` should be moments or distributions'
                    log.error(sss)
                    sys.exit()


        if inittype == 'moments':
            self.scheme.equilibrium(self._m)
            self.scheme.m2f(self._m, self._F)
        else:
            self.scheme.f2m(self._F, self._m)

        if not self.nv_on_beg:
            self._Fold[:] = self._F[:]

    def transport(self):
        t = time.time()
        self.scheme.transport(self._F)
        self.cpu_time['transport'] += time.time() - t

    def relaxation(self):
        t = time.time()
        self.scheme.relaxation(self._m)
        self.cpu_time['relaxation'] += time.time() - t

    def f2m(self):
        t = time.time()
        self.scheme.f2m(self._F, self._m)
        self.cpu_time['f2m_m2f'] += time.time() - t

    def m2f(self):
        t = time.time()
        self.scheme.m2f(self._m, self._F)
        self.cpu_time['f2m_m2f'] += time.time() - t

    def equilibrium(self):
        self.scheme.equilibrium(self._m)

    def boundary_condition(self):
        t = time.time()
        if self.dim == 1:
            # periodic for the moment
            log.debug("Boundary condition in 1D: only Neumann are implemented")
            if self.nv_on_beg:
                self._F[:,  0] = self._F[:,  1]
                self._F[:, -1] = self._F[:, -2]
            else:
                self._F[ 0, :] = self._F[ 1, :]
                self._F[-1, :] = self._F[-2, :]
        elif self.dim == 2:
            self.scheme.set_boundary_conditions(self._F, self._m, self.bc, self.nv_on_beg)
        else:
            log.error("Boundary conditions not yet implemented in 3D (maybe in another release)")
        self.cpu_time['boundary_conditions'] += time.time() - t

    def one_time_step(self):
        t = time.time()
        self.boundary_condition()

        if self.nv_on_beg:
            self.transport()
            self.f2m()
            self.relaxation()
            self.m2f()
        else:
            self._Fold[:] = self._F[:]
            self.scheme.onetimestep(self._m, self._F, self._Fold, self.domain.in_or_out, self.domain.valin)
            ftmp = self._Fold
            self._Fold = self._F
            self._F = ftmp

        self.t += self.dt
        self.nt += 1
        self.cpu_time['total'] += time.time() - t
        self.cpu_time['number_of_iterations'] += 1
        dummy = self.cpu_time['number_of_iterations']
        for n in self.domain.N:
            dummy *= n
        dummy /= self.cpu_time['total'] * 1.e6
        self.cpu_time['MLUPS'] = dummy

    def affiche_2D(self):
        fig = plt.figure(0,figsize=(8, 8))
        fig.clf()
        plt.ion()
        plt.imshow(np.float32(self.m[0][0][1:-1,1:-1].transpose()), origin='lower', cmap=cm.gray, interpolation='nearest')
        plt.title("Solution",fontsize=14)
        plt.draw()
        plt.hold(False)
        plt.ioff()
        plt.show()

    def affiche_1D(self):
        fig = plt.figure(0,figsize=(8, 8))
        fig.clf()
        plt.ion()
        plt.plot(self.domain.x[0][1:-1],self.m[0][0][1:-1])
        plt.title("Solution",fontsize=14)
        plt.draw()
        plt.hold(False)
        plt.ioff()
        #plt.show()

if __name__ == "__main__":
    pass
