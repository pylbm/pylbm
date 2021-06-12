# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
pylbm simulation
"""

import os
import sys
import logging
import types
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import mpi4py.MPI as mpi

from .domain import Domain
from .scheme import Scheme
from .boundary import Boundary
from . import utils
from .validator import validate
from .context import set_queue
from .generator import Generator
from .container import NumpyContainer, CythonContainer, LoopyContainer
from .algorithm import PullAlgorithm
from .monitoring import Monitor, monitor

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Simulation:
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
      a numpy array that contains the values of the distribution functions
      in each point
    m_halo : numpy array
      a numpy array that contains the values of the moments in each point
    F_halo : numpy array
      a numpy array that contains the values of the distribution functions
      in each point

    Examples
    --------

    see demo/examples/

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
    # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    def __init__(self, dico,
                 sorder=None, dtype='float64',
                 check_inverse=False,
                 initialize=True
                 ):
        validate(dico, __class__.__name__) #pylint: disable=undefined-variable

        self.domain = Domain(dico, need_validation=False)
        domain_size = mpi.COMM_WORLD.allreduce(sendobj=np.prod(self.domain.shape_in))
        Monitor.set_size(domain_size)

        self.scheme = Scheme(dico, check_inverse=check_inverse, need_validation=False)
        if self.domain.dim != self.scheme.dim:
            log.error('Solution: the dimension of the domain and of the scheme are not the same\n')
            sys.exit()

        self._update_m = True
        self.t = 0.
        self.nt = 0
        self.dt_ = self.domain.dx/self.scheme.la
        self.dim = self.domain.dim
        self.extra_parameters = {}

        codegen_dir, generate = None, True
        codegen_opt = dico.get('codegen_option', None)
        if codegen_opt:
            codegen_dir = os.path.realpath(codegen_opt['directory'])
            generate = codegen_opt.get('generate', True)

        self.generator = Generator(dico.get('generator', "CYTHON").upper(),
                                   codegen_dir,
                                   generate,
                                   dico.get('show_code', False))

        # FIXME remove that !!
        set_queue(self.generator.backend)

        self.container = self._get_container(sorder)
        if self.container.gpu_support:
            self.domain.in_or_out = self.container.move2gpu(self.domain.in_or_out)
            self.container.F.generate(self.generator)
            self.container.Fnew.generate(self.generator)
        sorder = self.container.sorder

        # Generate the numerical code for the LBM and for the boundary conditions
        self.algo = self._get_algorithm(dico, sorder)
        self.algo.generate()

        self.bc = Boundary(self.domain, self.generator, dico)
        for method in self.bc.methods:
            method.set_iload()
            method.generate(self.container.sorder)

        self.generator.compile()

        self.init_type = dico.get('inittype', 'moments')
        self.init_data = dico.get('init', None)

        self._need_init = True
        if initialize:
            self._initialize()

        log.info(self.__str__())

    def _initialize(self):
        # Initialize the solution and the rhs of boundary conditions
        self.initialization()
        for method in self.bc.methods:
            method.prepare_rhs(self)
            method.fix_iload()
            method.set_rhs()
            method.move2gpu()
        self._need_init = False

    def _get_container(self, sorder):
        container_type = {'NUMPY': NumpyContainer,
                          'CYTHON': CythonContainer,
                          'LOOPY': LoopyContainer
        }
        return container_type[self.generator.backend](self.domain, self.scheme, sorder)

    def _get_default_algo_settings(self):
        if self.generator.backend == 'NUMPY':
            return {'m_local': False, 'split': False, 'check_isfluid': False}
        else:
            return {'m_local': True, 'split': False, 'check_isfluid': False}

    def _get_algorithm(self, dico, sorder):
        algo_method = PullAlgorithm
        user_settings = {}
        dummy = dico.get('lbm_algorithm', None)
        if dummy:
            algo_method = dummy.get('name', PullAlgorithm)
            user_settings = dummy.get('settings', {})
        algo_settings = self._get_default_algo_settings()
        algo_settings.update(user_settings)

        return algo_method(self.scheme, sorder, self.generator, algo_settings)

    @property
    def dt(self):
        if isinstance(self.dt_, sp.Expr):
            subs = list(self.scheme.param.items()) + list(self.extra_parameters.items())
            self.dt_ = float(self.dt_.subs(subs))
            return self.dt_
        else:
            return self.dt_

    @utils.itemproperty
    def m_halo(self, i):
        """
        get the moment i on the whole domain with halo points.
        """
        if self._update_m:
            self._update_m = False
            self.f2m()
        return self.container.m[i]

    @m_halo.setter
    def m_halo(self, i, value):
        self._update_m = False
        self.container.m[i] = value

    @utils.itemproperty
    def m(self, i):
        """
        get the moment i in the interior domain.
        """
        if self._update_m:
            self._update_m = False
            self.f2m()

        return self.container.m._in(i) #pylint: disable=protected-access

    @utils.itemproperty
    def F_halo(self, i):
        """
        get the distribution function i on the whole domain with halo points.
        """
        return self.container.F[i]

    @F_halo.setter
    def F_halo(self, i, value):
        self._update_m = True
        self.container.F[i] = value

    @utils.itemproperty
    def F(self, i):
        """
        get the distribution function i in the interior domain.
        """
        return self.container.F._in(i) #pylint: disable=protected-access

    def __str__(self):
        from .utils import header_string
        from .jinja_env import env
        template = env.get_template('simulation.tpl')
        return template.render(header=header_string("Simulation information"),
                               simu=self)

    def __repr__(self):
        return self.__str__()

    @monitor
    def initialization(self):
        """
        initialize all the numy array with the initial conditions
        set the initial values to the numpy arrays _F and _m

        Parameters
        ----------

        dico : the dictionary with the `key:value` 'init'


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
        coords = np.meshgrid(*(c for c in self.domain.coords_halo), sparse=True, indexing='ij')

        if self.init_type == 'moments':
            array_to_init = self.container.m
        elif self.init_type == 'distributions':
            array_to_init = self.container.F
        else:
            sss = 'Error in the creation of the scheme: wrong dictionnary\n'
            sss += 'the key `inittype` should be moments or distributions'
            log.error(sss)
            sys.exit()

        if self.init_data is None:
            log.warning("You don't define initialization step for your conserved moments")
            return

        for k, v in self.init_data.items():
            if isinstance(v, tuple):
                f = v[0]
                extraargs = v[1] if len(v) == 2 else ()
                fargs = tuple(coords) + extraargs
                array_to_init[k] = f(*fargs)
            elif isinstance(v, types.FunctionType):
                fargs = tuple(coords)
                array_to_init[k] = v(*fargs)
            else:
                array_to_init[k] = v

        if self.init_type == 'moments':
            self.equilibrium()
            self.m2f()
        elif self.init_type == 'distributions':
            self.f2m()

        self.container.Fnew.array[:] = self.container.F.array[:]

    def transport(self, **kwargs):
        """
        compute the transport phase on distribution functions
        (the array _F is modified)
        """
        self.algo.call_function('transport', self, **kwargs)

    def relaxation(self, **kwargs):
        """
        compute the relaxation phase on moments
        (the array _m is modified)
        """
        self.algo.call_function('relaxation', self, **kwargs)

    def source_term(self, fraction_of_time_step=1., **kwargs):
        """
        compute the source term phase on moments
        (the array _m is modified)
        """
        self.algo.call_function('source_term', self, **kwargs)

    @monitor
    def f2m(self, **kwargs):
        """
        compute the moments from the distribution functions
        (the array _m is modified)
        """
        self.algo.call_function('f2m', self, **kwargs)

    @monitor
    def m2f(self, m_user=None, f_user=None, **kwargs):
        """
        compute the distribution functions from the moments
        (the array _F is modified)
        """
        self.algo.call_function('m2f', self, m_user, f_user, **kwargs)

    @monitor
    def equilibrium(self, m_user=None, **kwargs):
        """
        set the moments to the equilibrium values
        (the array _m is modified)

        Notes
        -----

        Another moments vector can be set to equilibrium values:
        use directly the method of the class Scheme
        """
        self.algo.call_function('equilibrium', self, m_user, **kwargs)

    @monitor
    def boundary_condition(self, **kwargs):
        """
        perform the boundary conditions

        Notes
        -----

        The array _F is modified in the phantom array (outer points)
        according to the specified boundary conditions.
        """
        f = self.container.F
        f.update()

        for method in self.bc.methods:
            method.update_feq(self)
            method.set_rhs()
            method.update(f, **kwargs)

    @monitor
    def one_time_step(self, **kwargs):
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
        if self._need_init:
            self._initialize()

        self._update_m = True # we recompute f so m will be not correct

        self.boundary_condition(**kwargs)

        self.algo.call_function('one_time_step', self, **kwargs)
        self.container.F, self.container.Fnew = self.container.Fnew, self.container.F

        self.t += self.dt
        self.nt += 1
