from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from six.moves import range
from six import string_types
import sys
import types
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
from . import utils
from .validate_dictionary import *
from .context import set_queue
from .generator import generator

from .logs import setLogger
from .storage import Array, Array_in, AOS, SOA

proto_simu = {
    'name':(type(None),) + string_types,
    'box':(is_dico_box,),
    'elements':(type(None), is_list_elem),
    'dim':(type(None), int),
    'space_step':(int, float, sp.Symbol),
    'scheme_velocity':(int, float, sp.Symbol),
    'parameters':(type(None), is_dico_sp_sporfloat),
    'schemes':(is_list_sch,),
    'relative_velocity': (type(None), is_list_sp_or_nb,),
    'boundary_conditions':(type(None), is_dico_bc),
    'generator':(type(None), is_generator),
    'show_code':(type(None), bool),
    'ode_solver':(type(None), is_ode_solver),
    'split_pattern': (type(None), is_list_string_or_tuple),
    'stability':(type(None), is_dico_stab),
    'consistency':(type(None), is_dico_cons),
    'inittype':(type(None),) + string_types,
}

class Simulation(object):
    """
    create a class simulation

    Parameters
    ----------

    dico : dictionary
    domain : object of class :py:class:`Domain<pylbm.domain.Domain>`, optional
    scheme : object of class :py:class:`Scheme<pylbm.scheme.Scheme>`, optional
    type :   optional argument (default value is 'float64')

    Attributes
    ----------

    dim : int
      spatial dimension
    type : float64
      the type of the values
    domain : :py:class:`Domain<pylbm.domain.Domain>`
      the domain given in argument
    scheme : :py:class:`Scheme<pylbm.scheme.Scheme>`
      the scheme given in argument
    m : numpy array
      a numpy array that contains the values of the moments in each point
    F : numpy array
      a numpy array that contains the values of the distribution functions in each point
    m_halo : numpy array
      a numpy array that contains the values of the moments in each point
    F_halo : numpy array
      a numpy array that contains the values of the distribution functions in each point

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
    :py:meth:`transport<pylbm.simulation.Simulation.transport>`,
    :py:meth:`relaxation<pylbm.simulation.Simulation.relaxation>`,
    :py:meth:`equilibrium<pylbm.simulation.Simulation.equilibrium>`,
    :py:meth:`f2m<pylbm.simulation.Simulation.f2m>`,
    :py:meth:`m2f<pylbm.simulation.Simulation.m2f>`,
    :py:meth:`boundary_condition<pylbm.simulation.Simulation.boundary_condition>`,
    and
    :py:meth:`one_time_step<pylbm.simulation.Simulation.one_time_step>`
    are just call of the methods of the class
    :py:class:`Scheme<pylbm.scheme.Scheme>`.
    """
    def __init__(self, dico, domain=None, scheme=None, sorder=None, dtype='float64', check_inverse=False):
        self.log = setLogger(__name__)
        self.type = dtype
        self.order = 'C'
        self._update_m = True

        self.log.info('Check the dictionary (by Simulation)')
        test, aff = validate(dico, proto_simu)
        if test:
            self.log.info(aff)
        else:
            self.log.error(aff)
            sys.exit()

        self.name = dico.get('name', None)

        self.log.info('Build the domain')
        try:
            if domain is not None:
                self.domain = domain
            else:
                self.domain = Domain(dico, verif=False)
        except KeyError:
            self.log.error('Error in the creation of the domain: wrong dictionnary')
            sys.exit()

        self.log.info('Build the scheme')
        try:
            if scheme is not None:
                self.scheme = scheme
            else:
                self.scheme = Scheme(dico, check_inverse=check_inverse)
        except KeyError:
            self.log.error('Error in the creation of the scheme: wrong dictionnary')
            sys.exit()

        self.t = 0.
        self.nt = 0
        self.dt = self.domain.dx/self.scheme.la
        try:
            assert self.domain.dim == self.scheme.dim
        except:
            self.log.error('Solution: the dimension of the domain and of the scheme are not the same\n')
            sys.exit()

        self.dim = self.domain.dim

        self.log.info('Build arrays')

        self.mpi_topo = self.domain.mpi_topo

        nv = self.scheme.stencil.nv_ptr[-1]
        nspace = self.domain.global_size
        vmax = self.domain.stencil.vmax

        self.generator = dico.get('generator', "CYTHON").upper()
        self.show_code = dico.get('show_code', False)
        set_queue(self.generator)
        self.gpu_support = True if self.generator=="LOOPY" else False

        if sorder is None:
            if self.generator == "NUMPY":
                self._m = SOA(nv, nspace, vmax, self.mpi_topo, gpu_support=self.gpu_support)
                self._F = SOA(nv, nspace, vmax, self.mpi_topo, gpu_support=self.gpu_support)
                #self._Fold = self._F
                sorder = [i for i in range(self.dim + 1)]
            else:
                self._m = AOS(nv, nspace, vmax, self.mpi_topo, gpu_support=self.gpu_support)
                self._F = AOS(nv, nspace, vmax, self.mpi_topo, gpu_support=self.gpu_support)
                sorder = [self.dim] + [i for i in range(self.dim)]
        else:
            self._m = Array(nv, nspace, vmax, sorder, self.mpi_topo, gpu_support=self.gpu_support)
            self._F = Array(nv, nspace, vmax, sorder, self.mpi_topo, gpu_support=self.gpu_support)

        self._m.set_conserved_moments(self.scheme.consm, self.domain.stencil.nv_ptr)
        self._F.set_conserved_moments(self.scheme.consm, self.domain.stencil.nv_ptr)

        if self.generator == "NUMPY":
            self._Fold = self._F
        else:
            self._Fold = Array(nv, nspace, vmax, sorder, self.mpi_topo, gpu_support=self.gpu_support)
            self._Fold.set_conserved_moments(self.scheme.consm, self.domain.stencil.nv_ptr)

        self.scheme.generate(self.generator, sorder, self.domain.valin)

        self.log.info('Build boundary conditions')

        if self.gpu_support:
            import pyopencl as cl
            import pyopencl.array
            from .context import queue

            self.domain.in_or_out = cl.array.to_device(queue, self.domain.in_or_out)  

        self.bc = Boundary(self.domain, dico)
        for method in self.bc.methods:
            method.set_iload()
            method.generate(sorder)

        generator.compile(backend=self.generator, verbose=self.show_code)

        self.log.info('Initialization')
        self.initialization(dico)
        for method in self.bc.methods:
            method.prepare_rhs(self)
            method.set_rhs()
            method.fix_iload()
            method.move2gpu()

        #computational time measurement
        self.cpu_time = {
            'relaxation':0.,
            'source_term':0.,
            'transport':0.,
            'f2m_m2f':0.,
            'boundary_conditions':0.,
            'total':0.,
            'number_of_iterations':0,
            'MLUPS':0.,
        }

    @utils.itemproperty
    def m_halo(self, i):
        """
        get the moment i on the whole domain with halo points.
        """
        if self._update_m:
            self._update_m = False
            self.f2m()
        return self._m[i]

    @m_halo.setter
    def m_halo(self, i, value):
        self._update_m = False
        self._m[i] = value

    @utils.itemproperty
    def m(self, i):
        """
        get the moment i in the interior domain.
        """
        if self._update_m:
            self._update_m = False
            self.f2m()
        #return self._m_in[i]
        return self._m._in(i)

    @m.setter
    def m(self, i, value):
        self._update_m = False
        self._m_in[i] = value

    @utils.itemproperty
    def F_halo(self, i):
        """
        get the distribution function i on the whole domain with halo points.
        """
        return self._F[i]

    @F_halo.setter
    def F_halo(self, i, value):
        self._update_m = True
        self._F[i] = value

    @utils.itemproperty
    def F(self, i):
        """
        get the distribution function i in the interior domain.
        """
        return self._F._in(i)
        #return self._F_in[i]

    @F.setter
    def F(self, i, value):
        self._update_m = True
        self._F_in[i] = value

    def __str__(self):
        s = "Simulation informations: "
        if self.name is not None:
            s += "[ " + self.name + " ]"
        s += '\n'
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
            tcut.append(ttot//u)
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
                s += '{0:2d}{1} '.format(int(tcut[k]), unity_name[k])
        s += '{0:2d}{1} '.format(int(tcut[-1]), unity_name[-1])
        if test_dummy:
            s += '{0:3d}ms'.format(tms) + ' '*13 + '*'
        else:
            s += ' '*18 + '*'
        if t['total'] == 0:
            ttotal = 1.e-15
        else:
            ttotal = t['total']
        s += '\n* ' + '-'*46 + ' *'
        s += '\n* relaxation         : {0:2d}%'.format(int(100*t['relaxation']/ttotal))
        s += ' '*23 + '*'
        s += '\n* source term        : {0:2d}%'.format(int(100*t['source_term']/ttotal))
        s += ' '*23 + '*'
        s += '\n* transport          : {0:2d}%'.format(int(100*t['transport']/ttotal))
        s += ' '*23 + '*'
        s += '\n* f2m_m2f            : {0:2d}%'.format(int(100*t['f2m_m2f']/ttotal))
        s += ' '*23 + '*'
        s += '\n* boundary conditions: {0:2d}%'.format(int(100*t['boundary_conditions']/ttotal))
        s += ' '*23 + '*'
        s += '\n' + '*'*50
        print(s)

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
        coords = np.meshgrid(*(c for c in self.domain.coords_halo), sparse=True, indexing='ij')

        if inittype == 'moments':
            array_to_init = self._m
        elif inittype == 'distributions':
            array_to_init = self._F
        else:
            sss = 'Error in the creation of the scheme: wrong dictionnary\n'
            sss += 'the key `inittype` should be moments or distributions'
            self.log.error(sss)
            sys.exit()

        for k, v in self.scheme.init.items():
            if isinstance(v, tuple):
                f = v[0]
                extraargs = v[1] if len(v) == 2 else ()
                fargs = tuple(coords) + extraargs
                array_to_init[k] = f(*fargs)
            else:
                array_to_init[k] = v

        if inittype == 'moments':
            self.scheme.equilibrium(self._m)
            self.scheme.m2f(self._m, self._F)
        elif inittype == 'distributions':
            self.scheme.f2m(self._F, self._m)

        self._Fold.array[:] = self._F.array[:]
        
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

    def source_term(self, fraction_of_time_step = 1.):
        """
        compute the source term phase on moments
        (the array _m is modified)
        """
        t = mpi.Wtime()
        self.scheme.source_term(self._m, self.t, fraction_of_time_step * self.dt, *self.domain.coords)
        self.cpu_time['source_term'] += mpi.Wtime() - t

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
        self.scheme.set_boundary_conditions(self._F, self._m, self.bc, self.mpi_topo)
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
        self._update_m = True # we recompute f so m will be not correct

        self.boundary_condition()

        self.scheme.onetimestep(self._m, self._F, self._Fold, self.domain.in_or_out, self.domain.valin, self.t, self.dt,
            *self.domain.coords)
        self._F, self._Fold = self._Fold, self._F

        self.t += self.dt
        self.nt += 1

if __name__ == "__main__":
    pass
