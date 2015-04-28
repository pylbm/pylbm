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

from .logs import setLogger

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

class Simulation:
    """
    create a class simulation

    Parameters
    ----------

    dico : dictionary
    domain : object of class :py:class:`Domain<pyLBM.domain.Domain>`, optional
    scheme : object of class :py:class:`Scheme<pyLBM.scheme.Scheme>`, optional
    type :   optional argument (default value is 'float64')

    Attributes
    ----------

    dim : int
      spatial dimension
    type : float64
      the type of the values
    domain : :py:class:`Domain<pyLBM.domain.Domain>`
      the domain given in argument
    scheme : :py:class:`Scheme<pyLBM.scheme.Scheme>`
      the scheme given in argument
    m : numpy array
      a numpy array that contains the values of the moments in each point
    F : numpy array
      a numpy array that contains the values of the distribution functions in each point

    Methods
    -------

    initialization :
      initialize all the arrays
    transport :
      compute the transport phase (modifies the array _F)
    relaxation :
      compute the relaxation phase (modifies the array _m)
    equilibrium :
      compute the equilibrium
    f2m :
      compute the moments _m from the distribution _F
    m2f :
      compute the distribution _F from the moments _m
    boundary_condition :
      compute the boundary conditions (modifies the array _F)
    one_time_step :
      compute a complet time step combining
      boundary_condition, transport, f2m, relaxation, m2f
    time_info :
      print informations about time

    Examples
    --------

    see demo/examples/

    Access to the distribution functions and the moments.

    In 1D::

    >>>F[n][k][i]
    >>>m[n][k][i]

    get the kth distribution function of the nth elementary scheme
    and the kth moment of the nth elementary scheme
    at the point x[0][i].

    In 2D::

    >>>F[n][k][j, i]
    >>>m[n][k][j, i]

    get the kth distribution function of the nth elementary scheme
    and the kth moment of the nth elementary scheme
    at the point x[0][i], x[1][j].


    Notes
    -----

    The methods
    :py:meth:`transport<pyLBM.simulation.Simulation.transport>`,
    :py:meth:`relaxation<pyLBM.simulation.Simulation.relaxation>`,
    :py:meth:`equilibrium<pyLBM.simulation.Simulation.equilibrium>`,
    :py:meth:`f2m<pyLBM.simulation.Simulation.f2m>`,
    :py:meth:`m2f<pyLBM.simulation.Simulation.m2f>`,
    :py:meth:`boundary_condition<pyLBM.simulation.Simulation.boundary_condition>`,
    and
    :py:meth:`one_time_step<pyLBM.simulation.Simulation.one_time_step>`
    are just call of the methods of the class
    :py:class:`Scheme<pyLBM.scheme.Scheme>`.
    """
    def __init__(self, dico, domain=None, scheme=None, dtype='float64'):
        self.log = setLogger(__name__)
        self.type = dtype
        self.order = 'C'

        self.log.info('Build the domain')
        try:
            if domain is not None:
                self.domain = domain
            else:
                self.domain = Domain(dico)
        except KeyError:
            self.log.error('Error in the creation of the domain: wrong dictionnary')
            sys.exit()

        self.log.info('Build the scheme')
        try:
            if scheme is not None:
                self.scheme = scheme
            else:
                self.scheme = Scheme(dico)
        except KeyError:
            self.log.error('Error in the creation of the scheme: wrong dictionnary')
            sys.exit()

        self.t = 0.
        self.nt = 0
        self.dt = self.domain.dx / self.scheme.la
        try:
            assert self.domain.dim == self.scheme.dim
        except:
            self.log.error('Solution: the dimension of the domain and of the scheme are not the same\n')
            sys.exit()

        self.dim = self.domain.dim


        self.log.info('Build arrays')
        #self.nv_on_beg = nv_on_beg
        self.nv_on_beg = self.scheme.nv_on_beg

        if self.nv_on_beg:
            msize = [self.scheme.stencil.nv_ptr[-1]] + self.domain.Na
            self._m = np.empty(msize, dtype=self.type, order=self.order)
            self._F = np.empty(msize, dtype=self.type, order=self.order)
        else:
            msize = self.domain.Na + [self.scheme.stencil.nv_ptr[-1]]
            self._m = np.empty(msize, dtype=self.type, order=self.order)
            self._F = np.empty(msize, dtype=self.type, order=self.order)
            self._Fold = np.empty(msize, dtype=self.type, order=self.order)

        self.interface = self.domain.geom.interface
        self.interface.set_subarray(self._F.shape, self.domain.stencil.vmax, self.nv_on_beg)

        self.log.info('Build boundary conditions')
        self.bc = Boundary(self.domain, dico)

        self.log.info('Initialization')
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
                if self.dim == 1:
                    return self._m[:, jj]
                elif self.dim == 2:
                    return self._m[:, :, jj]
                elif self.dim == 3:
                    return self._m[:, :, :, jj]
                else:
                    self.log.error('Bad value of spatial dimension dim = {0}'.format(self.dim))
        if self.nv_on_beg:
            return self._m[self.scheme.stencil.nv_ptr[i] + j]
        else:
            if self.dim == 1:
                return self._m[:, self.scheme.stencil.nv_ptr[i] + j]
            elif self.dim == 2:
                return self._m[:, :, self.scheme.stencil.nv_ptr[i] + j]
            elif self.dim == 3:
                return self._m[:, :, :, self.scheme.stencil.nv_ptr[i] + j]
            else:
                self.log.error('Bad value of spatial dimension dim = {0}'.format(self.dim))

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

    @property
    def mglobal(self):
        return self.interface.get_full(self._m, self.domain, self.nv_on_beg)

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

    def time_info(self):
        t = self.cpu_time
        # tranform the seconds into days, hours, minutes, seconds
        ttot = int(t['total'])
        tms = int(1000*(t['total'] - ttot))
        us = 1 # 1 second
        um = 60*us # 1 minute
        uh = 60*um # 1 hour
        ud = 24*uh # 1 day
        unity = [ud, uh, um, us]
        unity_name = ['d', 'h', 'm', 's']
        tcut = []
        for u in unity:
            tcut.append(ttot / u)
            ttot -= tcut[-1]*u
        #computational time measurement
        s = '*'*50
        s += '\n* Time informations' + ' '*30 + '*'
        s += '\n* ' + '-'*46 + ' *'
        s += '\n* MLUPS {0:5.1f}'.format(t['MLUPS']) + ' '*36 + '*'
        s += '\n* Number of iterations {0:10.3e}'.format(t['number_of_iterations'])
        s += ' '*16 + '*'
        s += '\n* Total time   '
        test_dummy = True
        for k in range(len(unity)-1):
            if (test_dummy and tcut[k] == 0):
                s += ' '*4
            else:
                test_dummy = False
                s += '{0:2d}{1} '.format(tcut[k], unity_name[k])
        s += '{0:2d}{1} '.format(tcut[-1], unity_name[-1])
        if test_dummy:
            s += '{0:3d}ms'.format(tms) + ' '*13 + '*'
        else:
            s += ' '*18 + '*'
        s += '\n* ' + '-'*46 + ' *'
        s += '\n* relaxation         : {0:2d}%'.format(int(100*t['relaxation']/t['total']))
        s += ' '*23 + '*'
        s += '\n* transport          : {0:2d}%'.format(int(100*t['transport']/t['total']))
        s += ' '*23 + '*'
        s += '\n* f2m_m2f            : {0:2d}%'.format(int(100*t['f2m_m2f']/t['total']))
        s += ' '*23 + '*'
        s += '\n* boundary conditions: {0:2d}%'.format(int(100*t['boundary_conditions']/t['total']))
        s += ' '*23 + '*'
        s += '\n' + '*'*50
        print s

    def initialization(self, dico):
        """
        initialize all the numy array with the initial conditions

        Parameters
        ----------

        dico : the dictionary with the `key:value` 'init'

        Returns
        -------

        set the initial values to the numpy arrays _F and _m

        Notes
        -----

        The initial values are set to _m, the array _F is then initialized
        with the equilibrium values.
        If the initial values have to be set to _F, use the optional
        `key:value` 'inittype' with the value 'distributions'
        (default value is set to 'moments').
        """
        # type of initialization
        # by default, the initialization is on the moments
        # else, it could be distributions
        inittype = dico.get('inittype', 'moments')
        if self.dim == 1:
            x = self.domain.x[0]
            coords = (x,)
        elif self.dim == 2:
            x = self.domain.x[0][:,np.newaxis]
            y = self.domain.x[1][np.newaxis, :]
            #x = self.domain.x[0][np.newaxis, :]
            #y = self.domain.x[1][: ,np.newaxis]
            coords = (x, y)

        schemes = dico['schemes']
        for ns, s in enumerate(schemes):
            for k, v in s['init'].iteritems():
                f = v[0]
                extraargs = v[1] if len(v) == 2 else ()
                fargs = coords + extraargs
                if self.nv_on_beg:
                    indices = self.scheme.stencil.nv_ptr[ns] + k
                else:
                    indices = (slice(None),)*self.dim + (self.scheme.stencil.nv_ptr[ns] + k,)
                if inittype == 'moments':
                    self._m[indices] = f(*fargs)
                elif inittype == 'distributions':
                    self._F[indices] = f(*fargs)
                else:
                    sss = 'Error in the creation of the scheme: wrong dictionnary\n'
                    sss += 'the key `inittype` should be moments or distributions'
                    self.log.error(sss)
                    sys.exit()


        if inittype == 'moments':
            self.scheme.equilibrium(self._m)
            self.scheme.m2f(self._m, self._F)
        elif inittype == 'distributions':
            self.scheme.f2m(self._F, self._m)

        if not self.nv_on_beg:
            self._Fold[:] = self._F[:]

    def transport(self):
        """
        compute the transport phase on distribution functions
        (the array _F is modified)
        """
        t = mpi.Wtime()
        self.scheme.transport(self._F)
        self.cpu_time['transport'] += mpi.Wtime() - t

    def relaxation(self):
        """
        compute the relaxation phase on moments
        (the array _m is modified)
        """
        t = mpi.Wtime()
        self.scheme.relaxation(self._m)
        self.cpu_time['relaxation'] += mpi.Wtime() - t

    def f2m(self):
        """
        compute the moments from the distribution functions
        (the array _m is modified)
        """
        t = mpi.Wtime()
        self.scheme.f2m(self._F, self._m)
        self.cpu_time['f2m_m2f'] += mpi.Wtime() - t

    def m2f(self):
        """
        compute the distribution functions from the moments
        (the array _F is modified)
        """
        t = mpi.Wtime()
        self.scheme.m2f(self._m, self._F)
        self.cpu_time['f2m_m2f'] += mpi.Wtime() - t

    def equilibrium(self):
        """
        set the moments to the equilibrium values
        (the array _m is modified)

        Notes
        -----

        Another moments vector can be set to equilibrium values:
        use directly the method of the class Scheme
        """
        self.scheme.equilibrium(self._m)

    def boundary_condition(self):
        """
        perform the boundary conditions

        Notes
        -----

        The array _F is modified in the phantom array (outer points)
        according to the specified boundary conditions.
        """
        t = mpi.Wtime()
        self.scheme.set_boundary_conditions(self._F, self._m, self.bc, self.interface, self.nv_on_beg)
        self.cpu_time['boundary_conditions'] += mpi.Wtime() - t

    def one_time_step(self):
        """
        compute one time step

        Notes
        -----

        Modify the arrays _F and _m in order to go further of dt.
        This function is equivalent to successively use

        - boundary_condition
        - transport
        - f2m
        - relaxation
        - m2f
        """
        t1 = mpi.Wtime()
        self.boundary_condition()

        if self.nv_on_beg:
            self.transport()
            self.f2m()
            self.relaxation()
            self.m2f()
        else:
            tloci = mpi.Wtime()
            self.scheme.onetimestep(self._m, self._F, self._Fold, self.domain.in_or_out, self.domain.valin)
            self._F, self._Fold = self._Fold, self._F
            tlocf = mpi.Wtime()
            self.cpu_time['transport'] += 0.5*(tlocf-tloci)
            self.cpu_time['relaxation'] += 0.5*(tlocf-tloci)
        t2 = mpi.Wtime()
        self.cpu_time['total'] += t2 - t1
        self.cpu_time['number_of_iterations'] += 1

        self.t += self.dt
        self.nt += 1
        dummy = self.cpu_time['number_of_iterations']
        for n in self.domain.Ng:
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
