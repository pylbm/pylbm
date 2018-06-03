from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys
import types
from six import string_types
from six.moves import range
import time

import numpy as np
import sympy as sp
from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
import copy
from textwrap import dedent

from .stencil import Stencil
from .validate_dictionary import *
from .generator import generator

from .logs import setLogger
import mpi4py.MPI as mpi

rel_ux, rel_uy, rel_uz = sp.symbols('rel_ux, rel_uy, rel_uz', real=True)

proto_sch = {
    'velocities': (is_list_int,),
    'conserved_moments': (sp.Symbol, is_list_symb) + string_types,
    'polynomials': (is_list_sp_or_nb,),
    'equilibrium': (type(None), is_list_sp_or_nb,),
    'feq': (type(None), tuple),
    'relaxation_parameters': (is_list_sp_or_nb,),
    'source_terms': (type(None), is_dico_sources),
    'init':(type(None), is_dico_init),
}

proto_sch_dom = {
    'velocities': (is_list_int,),
    'conserved_moments': (type(None), sp.Symbol, is_list_symb) + string_types,
    'polynomials': (type(None), is_list_sp_or_nb,),
    'equilibrium': (type(None), is_list_sp_or_nb,),
    'feq': (type(None), tuple),
    'relaxation_parameters': (type(None), is_list_sp_or_nb,),
    'source_terms': (type(None), is_dico_sources),
    'init':(type(None), is_dico_init),
}

proto_stab = {
    'linearization':(type(None), is_dico_sp_float),
    'test_monotonic_stability':(type(None), bool),
    'test_L2_stability':(type(None), bool),
}

proto_cons = {
    'order': (int,),
    'linearization':(type(None), is_dico_sp_sporfloat),
}

def alltogether(M):
    for i in range(M.shape[0]):
       for j in range(M.shape[1]):
            M[i, j] = M[i, j].expand().together().factor()

def allfactor(M):
    for i in range(M.shape[0]):
       for j in range(M.shape[1]):
            M[i, j] = M[i, j].factor()

def param_to_tuple(param):
    if param is not None:
        pk, pv = list(param.keys()), list(param.values())
    else:
        pk, pv = (), ()
    return pk, pv

class Scheme(object):
    """
    Create the class with all the needed informations for each elementary scheme.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`
      - dim : spatial dimension (optional if the `box` is given)
      - scheme_velocity : the value of the ratio space step over time step
        (la = dx / dt)
      - schemes : a list of dictionaries, one for each scheme
      - generator : a generator for the code, optional
        (see :py:class:`Generator <pylbm.generator.base.Generator>`)
      - ode_solver : a method to integrate the source terms, optional
        (see :py:class:`ode_solver <pylbm.generator.ode_schemes.ode_solver>`)
      - test_stability : boolean (optional)

    Notes
    -----

    Each dictionary of the list `schemes` should contains the following `key:value`

    - velocities : list of the velocities number
    - conserved moments : list of the moments conserved by each scheme
    - polynomials : list of the polynomial functions that define the moments
    - equilibrium : list of the values that define the equilibrium
    - relaxation_parameters : list of the value of the relaxation parameters
    - source_terms : dictionary do define the source terms (optional, see examples)
    - init : dictionary to define the initial conditions (see examples)

    If the stencil has already been computed, it can be pass in argument.

    Attributes
    ----------

    dim : int
      spatial dimension
    dx : double
      space step
    dt : double
      time step
    la : double
      scheme velocity, ratio dx/dt
    nscheme : int
      number of elementary schemes
    stencil : object of class :py:class:`Stencil <pylbm.stencil.Stencil>`
      a stencil of velocities
    P : list of sympy matrix
      list of polynomials that define the moments
    EQ : list of sympy matrix
      list of the equilibrium functions
    s  : list of list of doubles
      relaxation parameters
      (exemple: s[k][l] is the parameter associated to the lth moment in the kth scheme)
    M : sympy matrix
      the symbolic matrix of the moments
    Mnum : numpy array
      the numeric matrix of the moments (m = Mnum F)
    invM : sympy matrix
      the symbolic inverse matrix
    invMnum : numpy array
      the numeric inverse matrix (F = invMnum m)
    generator : :py:class:`Generator <pylbm.generator.base.Generator>`
      the used generator (
      :py:class:`NumpyGenerator<pylbm.generator.NumpyGenerator>`,
      :py:class:`CythonGenerator<pylbm.generator.CythonGenerator>`,
      ...)
    ode_solver : :py:class:`ode_solver <pylbm.generator.ode_schemes.ode_solver>`,
      the used ODE solver (
      :py:class:`explicit_euler<pylbm.generator.explicit_euler>`,
      :py:class:`heun<pylbm.generator.heun>`,
      ...)

    Examples
    --------

    see demo/examples/scheme/

    """
    def __init__(self, dico, stencil=None, check_inverse=False):
        self.log = setLogger(__name__)
        self.check_inverse = check_inverse
        # symbolic parameters
        self.param = dico.get('parameters', None)
        pk, pv = param_to_tuple(self.param)

        if stencil is not None:
            self.stencil = stencil
        else:
            self.stencil = Stencil(dico)
        self.dim = self.stencil.dim

        la = dico.get('scheme_velocity', None)
        if isinstance(la, (int, float)):
            self.la_symb = None
            self.la = la
        elif isinstance(la, sp.Symbol):
            self.la_symb = la
            self.la = float(la.subs(list(zip(pk, pv))))
        else:
            self.log.error("The entry 'scheme_velocity' is wrong.")
        dx = dico.get('space_step', None)
        if isinstance(dx, (int, float)):
            self.dx_symb = None
            self.dx = dx
        elif isinstance(dx, sp.Symbol):
            self.dx_symb = dx
            self.dx = float(dx.subs(list(zip(pk, pv))))
        else:
            self.dx = 1.
            s = "The value 'space_step' is not given or wrong.\n"
            s += "The scheme takes default value: dx = 1."
            self.log.warning(s)
        self.dt = self.dx / self.la

        # set relative velocity
        self.rel_vel = dico.get('relative_velocity', [0]*self.dim)
        self.backend = dico.get('backend', "cython").upper()

        # fix the variables of time and space
        self.vart, self.varX = None, [None, None, None]
        if self.param is not None:
            self.vart = self.param.get('time', None)
            self.varX[0] = self.param.get('space_x', None)
            self.varX[1] = self.param.get('space_y', None)
            self.varX[2] = self.param.get('space_z', None)
        if self.vart is None:
            self.vart = sp.Symbol('t')
        if self.varX[0] is None:
            self.varX[0] = sp.Symbol('X')
        if self.varX[1] is None:
            self.varX[1] = sp.Symbol('Y')
        if self.varX[2] is None:
            self.varX[2] = sp.Symbol('Z')

        self.nscheme = self.stencil.nstencils
        scheme = dico['schemes']
        if not isinstance(scheme, list):
            self.log.error("The entry 'schemes' must be a list.")

        def create_matrix(L):
            """
            convert a list of strings to a sympy Matrix.
            """
            def auto_moments(tokens, local_dict, global_dict):
                """
                if the user uses a string to describe the moments like
                'm[0][0]', this function converts it as Symbol('m[0][0]').
                This fix the problem of auto_symbol that doesn't support
                indexing.
                """
                result = []
                i = 0
                print('tokens ->', tokens)
                while(i < len(tokens)):
                    tokNum, tokVal = tokens[i]
                    if tokVal == 'm':
                        print(tokVal)
                        name = ''.join([val for n, val in tokens[i:i+7]])
                        result.extend([(1, 'Symbol'),
                                       (51, '('),
                                       (3, "'{0}'".format(name)),
                                       (51, ')')])
                        i += 7
                    else:
                        result.append(tokens[i])
                        i += 1
                return result
            res = []
            for l in L:
                if isinstance(l, string_types):
                    res.append(parse_expr(l, transformations=(auto_moments,) + standard_transformations))
                else:
                    res.append(l)
            return sp.Matrix(res)


        self._check_entry_size(scheme, 'polynomials')
        #self._check_entry_size(scheme, 'equilibrium')
        self._check_entry_size(scheme, 'relaxation_parameters')

        self.P = sp.Matrix([p for s in scheme for p in s['polynomials']])
        self._source_terms = [s.get('source_terms', None) for s in scheme]

        self.M = None
        self.invM = None
        self.Tu = None

        self.create_moments_matrices()

        # self.EQ = sp.Matrix([e for s in scheme for e in s['equilibrium']])
        self.s = sp.Matrix([r for s in scheme for r in s['relaxation_parameters']])
        self.s_non_swap = self.s.copy()

        eq = []
        for i, s in enumerate(scheme):
            feq = s.get('feq', None)
            meq = s.get('equilibrium', None)
            if feq and meq:
                self.log.error("Error in the creation of the scheme %d: you can have only 'feq' or 'equilibrium'"%i)
                sys.exit()
            if meq:
                eq.append(meq)
            if feq:
                sli = slice(self.stencil.nv_ptr[i],self.stencil.nv_ptr[i+1])
                meq_tmp = self.M[sli, sli]*feq[0](self.stencil.get_all_velocities(i), *feq[1])
                meq_tmp.simplify()
                eq.append([e for e in meq_tmp])
        self.EQ = sp.Matrix([e for sublist in eq for e in sublist])
        self.EQ_non_swap = self.EQ.copy()
        self.log.info("Equilibrium:\n" + sp.pretty(self.EQ))
        self.log.info("Matrix M to transform the distribution into the moments\n" + sp.pretty(self.M))
        self.log.info("Matrix M^(-1) to transform the moments into the distribution\n" + sp.pretty(self.invM))
        # put conserved moments at the beginning of P, EQ and s
        self.consm = self._get_conserved_moments(scheme)

        permutations = []
        for ic, c in enumerate(self.consm.values()):
            permutations.append([ic, c])

        for p in permutations:
            self.EQ.row_swap(p[0], p[1])
            self.s.row_swap(p[0], p[1])
            self.M.row_swap(p[0], p[1])
            self.invM.col_swap(p[0], p[1])
            self.Tu.row_swap(p[0], p[1])
            self.Tu.col_swap(p[0], p[1])
            self.Tmu.row_swap(p[0], p[1])
            self.Tmu.col_swap(p[0], p[1])

        for ic, c in enumerate(self.consm.keys()):
            self.consm[c] = ic

        self.init = self.set_initialization(scheme)

        #self._source_terms = dico.get('source_terms', None)
        # if self.source_terms is not None:

        #     self._source_terms = [create_matrix(s) for s in self.source_terms]
        #     for cm, icm in self.consm.items():
        #         for i, eq in enumerate(self._source_terms):
        #             for j, e in enumerate(eq):
        #                 if e is not None:
        #                     self._source_terms[i][j] = e.subs(cm, m[icm])
        #     self.ode_solver = dico.get('ode_solver', basic)()
        # else:
        #     self._source_terms = None
        #     self.ode_solver = None

        # generate the code
        # if self._source_terms is None:
        #     dummypattern = ['transport', 'relaxation']
        # else:
        #     dummypattern = ['transport', ('source_term', 0.5), 'relaxation', ('source_term', 0.5)]
        # self.pattern = dico.get('split_pattern', dummypattern)
        self.generator = dico.get('generator', "CYTHON").upper()
        ssss = "Generator used for the scheme functions:\n{0}\n".format(self.generator)
        #ssss += "with the pattern " + self.pattern.__str__() + "\n"
        self.log.info(ssss)

        self.bc_compute = True

        if self.check_inverse:
            self._check_inverse_of_Tu()

        # stability
        dicostab = dico.get('stability', None)
        if dicostab is not None:
            dico_linearization = dicostab.get('linearization', None)
            if dico_linearization is not None:
                self.list_linearization = []
                for cm, cv in dico_linearization.items():
                    icm = self.consm[cm]
                    self.list_linearization.append((m[icm[0]][icm[1]], cv))
            else:
                self.list_linearization = None
            self.compute_amplification_matrix_relaxation()
            Li_stab = dicostab.get('test_monotonic_stability', False)
            if Li_stab:
                if self.is_monotonically_stable():
                    print("The scheme is monotonically stable")
                else:
                    print("The scheme is not monotonically stable")
            L2_stab = dicostab.get('test_L2_stability', False)
            if L2_stab:
                if self.is_L2_stable():
                    print("The scheme is stable for the norm L2")
                else:
                    print("The scheme is not stable for the norm L2")

        # consistency
        dicocons = dico.get('consistency', None)
        if dicocons is not None:
            self.compute_consistency(dicocons)

    def _check_entry_size(self, schemes, key):
        for i, s in enumerate(schemes):
            ls = len(s[key])
            nv = self.stencil.nv[i]
            if ls != nv:
                self.log.error(dedent("""\
                               the size of the entry for the key {0} in the scheme {1}
                               has not the same size of the stencil {1}: {2}, {3}""".format(key, i, ls, nv)))

    def __str__(self):
        s = "Scheme informations\n"
        s += "\t spatial dimension: dim={0:d}\n".format(self.dim)
        s += "\t number of schemes: nscheme={0:d}\n".format(self.nscheme)
        s += "\t number of velocities:\n"
        for k in range(self.nscheme):
            s += "    Stencil.nv[{0:d}]=".format(k) + str(self.stencil.nv[k]) + "\n"
        s += "\t velocities value:\n"
        for k in range(self.nscheme):
            s+="    v[{0:d}] = ".format(k)
            for v in self.stencil.v[k]:
                s += v.__str__() + ', '
            s += '\n'
        s += "\t polynomials:\n"
        kl = 0
        for k in range(self.nscheme):
            s += "    P[{0:d}] = ".format(k)
            for l in range(self.stencil.nv[k]):
                s += self.P[kl].__str__() + ", "
                kl += 1
            s += "\n"
        s += "\t equilibria:\n"
        kl = 0
        for k in range(self.nscheme):
            s += "    EQ[{0:d}] = ".format(k)
            for l in range(self.stencil.nv[k]):
                s += self.EQ_non_swap[kl].__str__() + ", "
                kl += 1
            s += "\n"
        s += "\t relaxation parameters:\n"
        kl = 0
        for k in range(self.nscheme):
            s += "    s[{0:d}] = ".format(k)
            for l in range(self.stencil.nv[k]):
                s += self.s_non_swap[kl].__str__() + ", "
                kl += 1
            s += "\n"
        s += "\t moments matrices\n"
        s += "M      = " + self.M.__str__() + "\n"
        s += "M^(-1) = " + self.invM.__str__() + "\n"
        return s

    def create_moments_matrices(self):
        """
        Create the moments matrices M and M^{-1} used to transform the repartition functions into the moments

        Three versions of these matrices are computed:

          - a sympy version M and invM for each scheme
          - a numerical version Mnum and invMnum for each scheme
          - a global numerical version MnumGlob and invMnumGlob for all the schemes
        """
        compt=0
        M = []
        invM = []
        Mu = []
        Tu = []

        u_tild = sp.Matrix([rel_ux, rel_uy, rel_uz])

        if self.la_symb is not None:
            LA = self.la_symb
        else:
            LA = self.la

        for iv, v in enumerate(self.stencil.v):
            p = self.P[self.stencil.nv_ptr[iv] : self.stencil.nv_ptr[iv+1]]
            compt+=1
            lv = len(v)
            M.append(sp.zeros(lv, lv))
            Mu.append(sp.zeros(lv, lv))
            for i in range(lv):
                for j in range(lv):
                    sublist = [(str(self.varX[d]), sp.Integer(v[j].v[d])*LA) for d in range(self.dim)]
                    M[-1][i, j] = p[i].subs(sublist)

                    if self.rel_vel != [0]*self.dim:
                        sublist = [(str(self.varX[d]), v[j].v[d]*LA - u_tild[d]) for d in range(self.dim)]
                        Mu[-1][i, j] = p[i].subs(sublist)

            invM.append(M[-1].inv())
            Tu.append(Mu[-1]*invM[-1])

        gshape = (self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1])
        self.Tu = sp.eye(gshape[0])
        self.M = sp.zeros(*gshape)
        self.invM = sp.zeros(*gshape)

        try:
            for k in range(self.nscheme):
                nvk = self.stencil.nv[k]
                for i in range(nvk):
                    for j in range(nvk):
                        index = self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j
                        self.M[index] = M[k][i, j]
                        self.invM[index] = invM[k][i, j]

                        if self.rel_vel != [0]*self.dim:
                            self.Tu[index] = Tu[k][i, j]
        except TypeError:
            self.log.error("Unable to convert to float the expression {0} or {1}.\nCheck the 'parameters' entry.".format(self.M[k][i, j], self.invM[k][i, j]))
            sys.exit()

        alltogether(self.Tu)
        alltogether(self.M)
        alltogether(self.invM)
        # compute the inverse of self.Tu by formula T(u)T(-u)=Id
        #self.Tmu = sp.eye(gshape[0])
        self.Tmu = self.Tu.subs(list(zip(u_tild, -u_tild)))
        # for k in range(gshape[0]):
        #     for l in range(gshape[0]):
        #         self.Tmu[k,l] = self.Tu[k,l].subs(list(zip(u_tild, -u_tild)))

    def _check_inverse_of_Tu(self):
        # verification
        res = self.Tu*self.Tmu
        alltogether(res)
        gshape = self.stencil.nv_ptr[-1]
        test = res == sp.eye(gshape)
        if not test:
            self.log.warning("The property on the translation matrix is not verified\n T(u) * T(-u) is not identity !!!")

    def _check_inverse(self, M, invM, matrix_name):
        # verification
        gshape = self.stencil.nv_ptr[-1]
        dummy = M*invM
        alltogether(dummy)
        test = dummy == sp.eye(gshape)
        if not test:
            self.log.warning("Problem {name} * inv{name} is not identity !!!".format(name=matrix_name))

    def _get_conserved_moments(self, scheme):
        """
        return conserved moments and their indices in the scheme entry.

        Parameters
        ----------

        scheme : dictionnary that describes the LBM schemes

        Returns
        -------

        consm : dictionnary where the keys are the conserved moments and
                the values their indices in the LBM schemes.
        """
        from collections import OrderedDict
        consm_tmp = [s.get('conserved_moments', None) for s in scheme]
        consm = OrderedDict()

        for i in range(len(self.stencil.nv_ptr)-1):
            leq = self.EQ[self.stencil.nv_ptr[i]:self.stencil.nv_ptr[i+1]]
            cm_ieq = consm_tmp[i]
            if cm_ieq is not None:
                if isinstance(cm_ieq, sp.Symbol):
                    consm[cm_ieq] = self.stencil.nv_ptr[i] + leq.index(cm_ieq)
                else:
                    for c in cm_ieq:
                        consm[c] = self.stencil.nv_ptr[i] + leq.index(c)

        # for s in scheme:
        #     tmp = s.get('conserved_moments', None)
        #     if s.get('conserved_moments', None):
        #         if isinstance(tmp, (list, tuple)):
        #             consm_tmp += tmp
        #         else:
        #             consm_tmp.append(tmp)

        # leq = self.EQ.tolist()

        # consm = {}
        # for c in consm_tmp:
        #     consm[c] = leq.index([c])

        # consm_tmp = [s.get('conserved_moments', None) for s in scheme]

        # def find_indices(ieq, list_eq, c):
        #     if [c] in leq:
        #         ic = (ieq, leq.index([c]))
        #         if isinstance(c, str):
        #             cm = parse_expr(c)
        #         else:
        #             cm = c
        #         return ic, cm

        # # find the indices of the conserved moments in the equilibrium equations
        # for ieq, eq in enumerate(self.EQ):
        #     leq = eq.tolist()
        #     cm_ieq = consm_tmp[ieq]
        #     if cm_ieq is not None:
        #         if isinstance(cm_ieq, sp.Symbol):
        #             ic, cm = find_indices(ieq, leq, cm_ieq)
        #             consm[cm] = ic
        #         else:
        #             for c in cm_ieq:
        #                 ic, cm = find_indices(ieq, leq, c)
        #                 consm[cm] = ic
        return consm

    def _get_indices_cons_noncons(self):
        """
        return the list of the conserved moments and the list of the non conserved moments

        Returns
        -------

        l_cons : the list of the indices of the conserved moments
        l_noncons : the list of the indices of the non conserver moments
        """

        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        l_cons = [[] for n in nv]
        l_noncons = [list(range(n)) for n in nv]
        for vk in list(self.consm.values()):
            l_cons[vk[0]].append(vk[1])
            l_noncons[vk[0]].remove(vk[1])
        for n in range(ns):
            l_cons[n].sort()
            l_noncons[n].sort()
        return l_cons, l_noncons

    def set_initialization(self, scheme):
        """
        set the initialization functions for the conserved moments.

        Parameters
        ----------

        scheme : dictionnary that describes the LBM schemes

        Returns
        -------

        init : dictionnary where the keys are the indices of the conserved moments and the values must be
           - a constant (int or float)
           - a tuple of size 2 that describes a function and its extra args

        """
        init = {}
        for ns, s in enumerate(scheme):
            init_scheme = s.get('init', None)
            if init_scheme is None:
                self.log.warning("You don't define initialization step for your conserved moments")
                continue
            for k, v in s['init'].items():

                try:
                    if isinstance(k, str):
                        indices = self.consm[parse_expr(k)]
                    elif isinstance(k, sp.Symbol):
                        indices = self.consm[k]
                    elif isinstance(k, int):
                        indices = (ns, k)
                    else:
                        raise ValueError

                    init[indices] = v

                except ValueError:
                    sss = 'Error in the creation of the scheme: wrong dictionnary\n'
                    sss += 'the key `init` should contain a dictionnary with'
                    sss += '   key: the moment to init'
                    sss += '        should be the name of the moment as a string or'
                    sss += '        a sympy Symbol or an integer'
                    sss += '   value: the initial value'
                    sss += '        should be a constant, a tuple with a function'
                    sss += '        and extra args or a lambda function'
                    self.log.error(sss)
                    sys.exit()
        return init

    def set_source_terms(self, scheme):
        """
        set the source terms functions for the conserved moments.

        Parameters
        ----------

        scheme : dictionnary that describes the LBM schemes

        Returns
        -------

        source_terms : dictionnary where the keys are the indices of the conserved moments
        and the values must be a sympy expression or None

        """
        source_terms = []
        is_empty = True
        for ns, s in enumerate(scheme):
            source_terms.append([None]*self.stencil.nv[ns]) # by default no source term
            source_scheme = s.get('source_terms', None)
            if source_scheme is not None:
                for k, v in s['source_terms'].items():
                    try:
                        if isinstance(k, str):
                            indices = self.consm[parse_expr(k)]
                        elif isinstance(k, sp.Symbol):
                            indices = self.consm[k]
                        elif isinstance(k, int):
                            indices = (ns, k)
                        else:
                            raise ValueError
                    except ValueError:
                        sss = 'Error in the creation of the scheme: wrong dictionnary\n'
                        sss += 'the key `source_terms` should contain a dictionnary with'
                        sss += '   key: the moment concerned'
                        sss += '        should be the name of the moment as a string or'
                        sss += '        a sympy Symbol or an integer'
                        sss += '   value: the value of the source term'
                        sss += '        should be a float or a sympy expression'
                        self.log.error(sss)
                        sys.exit()
                    source_terms[-1][indices[1]] = v
                    is_empty = False
        if is_empty:
            return None
        else:
            return source_terms

    def generate(self, backend, sorder, valin):
        """
        Generate the code by using the appropriated generator

        Notes
        -----

        The code can be viewed. If S is the scheme

        >>> print(S.generator.code)
        """
        pk, pv = param_to_tuple(self.param)

        # if self._source_terms is not None:
        #     ST = copy.deepcopy(self._source_terms)
        #     for i, sti in enumerate(ST):
        #         for j, stij in enumerate(sti):
        #             if stij is not None:
        #                 ST[i][j] = stij.subs(list(zip(pk, pv)))
        #     dicoST = {'ST':ST,
        #               'vart':self.vart,
        #               'varx':self.varX[0],
        #               'vary':self.varX[1],
        #               'varz':self.varX[2],
        #               'ode_solver':self.ode_solver}
        # else:
        #     dicoST = None

        ns = int(self.stencil.nv_ptr[-1])
        mv = sp.MatrixSymbol('m', ns, 1)

        subs_param = list(zip(pk, pv))
        subs_moments = list(zip(self.consm.keys(), [mv[int(i),0] for i in self.consm.values()]))

        eq = self.EQ.subs(subs_moments)
        s = self.s.subs(subs_moments + subs_param)
        alltogether(eq)
        alltogether(s)

        # create source terms equations using Euler explicit scheme
        source_eq = [m for m in mv]
        dt = symbols('dt')
        for source in self._source_terms:
            if source:
                for k, v in source.items():
                    lhs = k.subs(subs_moments)
                    rhs = lhs + .5*dt*v.subs(subs_moments)
                    source_eq[lhs.i] = rhs
        source_eq = sp.Matrix(source_eq)

        if self.rel_vel != [0]*self.dim:
            list_rel_vel = [rel_ux, rel_uy, rel_uz][:self.dim]

            Mu = self.Tu * self.M
            invMu = self.invM * self.Tmu
        else:
            Mu = self.M
            invMu = self.invM
            list_rel_vel = []

        # fix execution time
        if self.check_inverse:
            self._check_inverse(self.M, self.invM, 'M')
            self._check_inverse(Mu, invMu, 'Mu')

        M = self.M.subs(subs_param)
        invM = self.invM.subs(subs_param)
        Mu = Mu.subs(subs_param)
        invMu = invMu.subs(subs_param)

        alltogether(M)
        alltogether(invM)
        alltogether(Mu)
        alltogether(invMu)

        from .generator import For, If
        from .symbolic import nx, ny, nz, nv, indexed, space_loop

        iloop = space_loop([(0, nx), (0, ny), (0, nz)], permutation=sorder) # loop over all spatial points
        m = indexed('m', [ns, nx, ny, nz], index=[nv] + iloop, ranges=range(ns), permutation=sorder)
        f = indexed('f', [ns, nx, ny, nz], index=[nv] + iloop, ranges=range(ns), permutation=sorder)

        # WARNING: (relative velocties)
        # the moments in the functions f2m, m2f, and equilibrium
        # are the real moments even if the scheme uses a relative velocity

        # add the function f2m as m = M f
        generator.add_routine(('f2m', For(iloop, Eq(m, M*f))), settings={"prefetch":[f[0]]})
        # add the function m2f as f = M^(-1) m
        generator.add_routine(('m2f', For(iloop, Eq(f, invM*m))), settings={"prefetch":[m[0]]})
        # add the function equilibrium
        dummy = eq.subs(list(zip(mv, m)) + subs_param)
        alltogether(dummy)
        generator.add_routine(('equilibrium', For(iloop, Eq(m, dummy))))

        # fix: set loop with vmax -> DONE ?
        vmax = [0]*3
        vmax[:self.dim] = self.stencil.vmax
        iloop = space_loop([(vmax[0], nx-vmax[0]),
                            (vmax[1], ny-vmax[1]),
                            (vmax[2], nz-vmax[2])], permutation=sorder)
        f = indexed('f', [ns, nx, ny, nz], index=[nv] + iloop, list_ind=self.stencil.get_all_velocities(), permutation=sorder)
        f_new = indexed('f_new', [ns, nx, ny, nz], index=[nv] + iloop, ranges=range(ns), permutation=sorder)
        in_or_out = indexed('in_or_out', [ns, nx, ny, nz], permutation=sorder, remove_ind=[0])

        if backend.upper() == "NUMPY":
            ################## FIX
            #self.log.error("NUMPY generator not allowed in this version with relative velocities")
            m = indexed('m', [ns, nx, ny, nz], index=[nv] + iloop, ranges=range(ns), permutation=sorder)
            dummy = eq.subs(list(zip(mv, m)) + subs_param).expand()
            source_eq = source_eq.subs(list(zip(mv, m)) + subs_param).expand()
            # FIX: have a unique version for source terms. The issue here is that sympy
            #      return True for Eq(m, source_eq) if there are no source terms which is not
            #      a valid expression for codegen (could be fix if we can use Assignment instead of Eq)
            if Eq(m, source_eq) == True:
                generator.add_routine(('one_time_step', For(iloop,
                                                            [
                                                                Eq(m, Mu*f),  # transport + f2m
                                                                Eq(m, (sp.ones(*s.shape) - s).multiply_elementwise(sp.Matrix(m)) + s.multiply_elementwise(dummy)), # relaxation
                                                                Eq(f_new, invMu*m), # m2f + update f
                                                            ]
                                                            )
                                                        ))
            else:
                generator.add_routine(('one_time_step', For(iloop,
                                                            [
                                                                Eq(m, Mu*f),  # transport + f2m
                                                                Eq(m, sp.Matrix(source_eq)), # source terms
                                                                Eq(m, (sp.ones(*s.shape) - s).multiply_elementwise(sp.Matrix(m)) + s.multiply_elementwise(dummy)), # relaxation
                                                                Eq(m, sp.Matrix(source_eq)), # source terms
                                                                Eq(f_new, invMu*m), # m2f + update f
                                                            ]
                                                            )
                                                        ))
        else:
            # build the equations defining the relative velocities
            nconsm = len(self.consm)
            brv = [Eq(mv[:nconsm, 0], sp.Matrix((M*f)[:nconsm]))]
            if self.rel_vel != [0]*self.dim:
                rel_vel = sp.Matrix(self.rel_vel).subs(subs_moments)
                for i in range(self.dim):
                    brv.append(Eq(list_rel_vel[i], rel_vel[i]))
                # build the equilibrium
                #dummy = self.Tu.subs(subs_param)*eq.subs(subs_param)
                dummy = self.Tu*eq
                dummy = dummy.subs(subs_param)
                alltogether(dummy)
            else:
                dummy = eq.subs(subs_param)

            source_eq = source_eq.subs(subs_param).expand()

            if all([src_t is None for src_t in self._source_terms]):
                generator.add_routine(('one_time_step',
                                        For(iloop,
                                            If( (Eq(in_or_out, valin),
                                                brv + # build relative velocity
                                                [Eq(mv, sp.Matrix(Mu*f)), # relative non conserved moments
                                                Eq(mv, (sp.ones(*s.shape) - s).multiply_elementwise(sp.Matrix(mv)) + s.multiply_elementwise(dummy)), # relaxation
                                                Eq(f_new, invMu*mv), # m2f + update f_new
                                                ]) )
                                            )
                                        ), local_vars = [mv] + list_rel_vel, settings={"prefetch":[f[0]]})
            else:
                generator.add_routine(('one_time_step',
                                        For(iloop,
                                            If( (Eq(in_or_out, valin),
                                                brv + # build relative velocity
                                                [Eq(mv, sp.Matrix(Mu*f)), # relative non conserved moments
                                                Eq(mv, source_eq), # source terms
                                                Eq(mv, (sp.ones(*s.shape) - s).multiply_elementwise(sp.Matrix(mv)) + s.multiply_elementwise(dummy)), # relaxation
                                                Eq(mv, source_eq),  # source terms
                                                Eq(f_new, invMu*mv), # m2f + update f_new
                                                ]) )
                                            )
                                        ), local_vars = [mv] + list_rel_vel, settings={"prefetch":[f[0]]})

            ## FIX: relative velocity
            # generator.add_routine(('one_time_step',
            #                           For(iloop,
            #                               If( (Eq(in_or_out, valin),
            #                                   brv + # build relative velocity
            #                                   [Eq(mv[nconsm:,0], sp.Matrix((Mu*f)[nconsm:])), # relative non conserved moments
            #                                   Eq(mv, source_eq), # source terms
            #                                   Eq(mv, (sp.ones(*s.shape) - s).multiply_elementwise(sp.Matrix(mv)) + s.multiply_elementwise(dummy)), # relaxation
            #                                   Eq(mv[:nconsm, 0], sp.Matrix((Mu*f)[:nconsm])), #relative conserved moments
            #                                   Eq(mv, source_eq),  # source terms
            #                                   Eq(f_new, invMu*mv), # m2f + update f_new
            #                                   ]) )
            #                               )
            #                           ), local_vars = [mv] + list_rel_vel, settings={"prefetch":[f[0]]})

    def m2f(self, mm, ff):
        """ Compute the distribution functions f from the moments m """
        from .symbolic import call_genfunction

        nx = mm.nspace[0]
        if self.dim > 1:
            ny = mm.nspace[1]
        if self.dim > 2:
            nz = mm.nspace[2]

        m = mm.array
        f = ff.array

        args = locals()
        call_genfunction(generator.module.m2f, args)

    def f2m(self, ff, mm):
        """ Compute the moments m from the distribution functions f """
        from .symbolic import call_genfunction

        nx = mm.nspace[0]
        if self.dim > 1:
            ny = mm.nspace[1]
        if self.dim > 2:
            nz = mm.nspace[2]

        m = mm.array
        f = ff.array

        args = locals()
        call_genfunction(generator.module.f2m, args)

    def transport(self, f):
        """ The transport phase on the distribution functions f """
        mod.transport(f.array)

    def equilibrium(self, mm):
        """ Compute the equilibrium """
        from .symbolic import call_genfunction

        nx = mm.nspace[0]
        if self.dim > 1:
            ny = mm.nspace[1]
        if self.dim > 2:
            nz = mm.nspace[2]

        m = mm.array

        args = locals()
        call_genfunction(generator.module.equilibrium, args)

    def relaxation(self, m):
        """ The relaxation phase on the moments m """
        mod = self.generator.get_module()
        mod.relaxation(m.array)

    def source_term(self, m, tn=0., dt=0., x=0., y=0., z=0.):
        """ The integration of the source term on the moments m """
        mod = self.generator.get_module()
        mod.source_term(m.array, tn, dt, x, y, z)

    def onetimestep(self, mm, ff, ff_new, in_or_out, valin, tn=0., dt=0., x=0., y=0., z=0.):
        """ Compute one time step of the Lattice Boltzmann method """
        from .symbolic import call_genfunction

        nx = mm.nspace[0]
        if self.dim > 1:
            ny = mm.nspace[1]
        if self.dim > 2:
            nz = mm.nspace[2]

        m = mm.array
        f = ff.array
        f_new = ff_new.array

        args = locals()
        call_genfunction(generator.module.one_time_step, args)

    def set_boundary_conditions(self, f, m, bc, interface):
        """
        Apply the boundary conditions

        Parameters
        ----------

        f : numpy array
          the array of the distribution functions
        m : numpy array
          the array of the moments
        bc : :py:class:`pylbm.boundary.Boundary`
          the class that contains all the informations needed
          for the boundary conditions

        Returns
        -------

        Modify the array of the distribution functions f in the phantom border area
        according to the labels. In the direction parallel to the bounday, N denotes
        the number of inner points, phantom cells are added to take into account
        the boundary conditions.

        Notes
        -----

        If n is the number of outer cells on each bound and N the number of inner cells,
        the following representation could be usefull (Na = N+2*n)

         +---------------+----------------+-----------------+
         | n outer cells | N inner cells  | n outer cells   |
         +===============+================+=================+
         |               | 0 ...  N-1     |                 |
         +---------------+----------------+-----------------+
         | 0  ...  n-1   | n ... N+n-1    | N+n  ... Na-1   |
         +---------------+----------------+-----------------+

        """
        f.update()

        for method in bc.methods:
            method.update(f)

    def compute_amplification_matrix_relaxation(self):
        """
        compute the amplification matrix of the relaxation.

        Returns
        -------

        amplification_matrix_relaxation : numpy array
          the matrix of the relaxation in the frame of the distribution functions

        Notes
        -----

        The output matrix corresponds to the linear operator involved
        in the relaxation phase. If the equilibrium is not a linear combination
        of the conserved moments, a linearization is done arround a given state.
        """
        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        nvtot = sum(nv)
        # matrix of the f2m and m2f transformations
        M = np.zeros((nvtot, nvtot))
        iM = np.zeros((nvtot, nvtot))
        # matrix of the relaxation parameters
        R = np.zeros((nvtot, nvtot))
        # matrix of the equilibrium
        E = np.zeros((nvtot, nvtot))
        k = 0
        for n in range(ns):
            l = nv[n]
            M[k:k+l, k:k+l] = self.Mnum[n]
            iM[k:k+l, k:k+l] = self.invMnum[n]
            R[k:k+l, k:k+l] = np.diag(self.s[n])
            k += l
        k = 0

        pk, pv = param_to_tuple(self.param)
        m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in range(self.stencil.unvtot)] for i in range(len(self.EQ))]
        for n in range(ns):
            for i in range(nv[n]):
                eqi = self._EQ[n][i].subs(list(zip(pk, pv)))
                if str(eqi) != "u[%d][%d]"%(n, i):
                    l = 0
                    for mm in range(ns):
                        for j in range(nv[mm]):
                            dummy = sp.diff(eqi, m[mm][j])
                            if self.list_linearization is not None:
                                dummy = dummy.subs(self.list_linearization)
                            E[k+i, l+j] = dummy
                        l += nv[mm]
            k += nv[n]
        C = np.dot(R, E - np.eye(nvtot))
        # global amplification matrix for the relaxation
        self.amplification_matrix_relaxation = np.eye(nvtot) + np.dot(iM, np.dot(C, M))

    def compute_amplification_matrix(self, wave_vector):
        """
        compute the amplification matrix of one time step of the scheme
        for the given wave vector.

        Returns
        -------

        amplification_matrix : numpy array
          the matrix of one time step of the sheme in the frame of the distribution functions

        Notes
        -----

        The output matrix corresponds to the linear operator involved
        in the relaxation phase. If the equilibrium is not a linear combination
        of the conserved moments, a linearization is done arround a given state.
        """
        Jr = self.amplification_matrix_relaxation
        # matrix of the transport phase
        q = Jr.shape[0]
        J = np.zeros((q, q), dtype='complex128')
        k = 0
        for n in range(self.stencil.nstencils):
            for i in range(self.stencil.nv[n]):
                vi = [self.stencil.vx[n][i],
                      self.stencil.vy[n][i],
                      self.stencil.vz[n][i]]
                J[k+i, :] = np.exp(1j*sum([a*b for a, b in zip(wave_vector, vi)])) * Jr[k+i, :]
            k += self.stencil.nv[n]
        return J

    def vp_amplification_matrix(self, wave_vector):
        """
        compute the eigenvalues of the amplification matrix
        for a given wave vector.

        Returns
        -------

        eigenvalues : numpy array
          the complex eigenvalues computed by numpy.linalg.eig
        """
        return np.linalg.eig(self.compute_amplification_matrix(wave_vector))[0]

    def is_L2_stable(self, Nk = 101):
        """
        test the L2 stability of the scheme

        Notes
        -----

        If the equilibrium is not a linear combination
        of the conserved moments,
        a linearization is done arround a given state.

        The test is performed for Nk^d (default value Nk=101) wave vectors
        uniformly distributed in [0,2pi]^d where d is the spatial dimension.

        """
        R = 1.
        vk = np.linspace(0., 2*np.pi, Nk)
        if self.dim == 1:
            for i in range(vk.size):
                kx = vk[i]
                vp = self.vp_amplification_matrix((kx, ))
                rloc = max(abs(vp))
                if rloc > R+1.e-14:
                    return False
        elif self.dim == 2:
            for i in range(vk.size):
                kx = vk[i]
                for j in range(vk.size):
                    ky = vk[j]
                    vp = self.vp_amplification_matrix((kx, ky))
                    rloc = max(abs(vp))
                    if rloc > R+1.e-14:
                        return False
        elif self.dim == 3:
            for i in range(vk.size):
                kx = vk[i]
                for j in range(vk.size):
                    ky = vk[j]
                    for k in range(vk.size):
                        kz = vk[k]
                        vp = self.vp_amplification_matrix((kx, ky, kz))
                        rloc = max(abs(vp))
                        if rloc > R+1.e-14:
                            return False
        else:
            self.log.warning("dim should be in [1, 3] for the scheme")
        return True

    def is_monotonically_stable(self):
        """
        test the monotonical stability of the scheme.

        Notes
        -----


        """
        if np.min(self.amplification_matrix_relaxation) < 0:
            return False
        else:
            return True

    def compute_consistency(self, dicocons):
        """
        compute the consistency of the scheme.

        FIX: documentation
        """
        t0 = mpi.Wtime()
        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        nvtot = self.stencil.nv_ptr[-1] # total number of velocities (with repetition)
        N = len(self.consm) # number of conserved moments
        time_step = sp.Symbol('h')
        drondt = sp.Symbol("dt") # time derivative
        drondx = [sp.Symbol("dx"), sp.Symbol("dy"), sp.Symbol("dz")] # spatial derivatives
        if self.la_symb is not None:
            LA = self.la_symb
        else:
            LA = self.la

        m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in range(nvtot)] for i in range(ns)]
        order = dicocons['order']
        if order<1:
            order = 1
        dico_linearization = dicocons.get('linearization', None)
        if dico_linearization is not None:
            self.list_linearization = []
            for cm, cv in dico_linearization.items():
                icm = self.consm[cm]
                self.list_linearization.append((m[icm[0]][icm[1]], cv))
        else:
            self.list_linearization = None

        M = sp.zeros(nvtot, nvtot)
        invM = sp.zeros(nvtot, nvtot)
        il = 0
        for n in range(ns):
            for k in self.ind_cons[n]:
                M[il,self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]] = self.M[n][k,:]
                invM[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1],il] = self.invM[n][:,k]
                il += 1
        for n in range(ns):
            for k in self.ind_noncons[n]:
                M[il,self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]] = self.M[n][k,:]
                invM[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1],il] = self.invM[n][:,k]
                il += 1
        v = self.stencil.get_all_velocities().transpose()

        # build the matrix of equilibrium
        Eeq = sp.zeros(nvtot, nvtot)
        il = 0
        for n_i in range(ns):
            for k_i in self.ind_cons[n_i]:
                Eeq[il, il] = 1
                ## the equilibrium value of the conserved moments is itself
                #eqk = self._EQ[n_i][k_i]
                #ic = 0
                #for n_j in xrange(len(self.ind_cons)):
                #    for k_j in self.ind_cons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                #for n_j in xrange(len(self.ind_noncons)):
                #    for k_j in self.ind_noncons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                il += 1
        for n_i in range(ns):
            for k_i in self.ind_noncons[n_i]:
                eqk = self._EQ[n_i][k_i]
                ic = 0
                for n_j in range(ns):
                    for k_j in self.ind_cons[n_j]:
                        dummy = sp.diff(eqk, m[n_j][k_j])
                        if self.list_linearization is not None:
                            dummy = dummy.subs(self.list_linearization)
                        Eeq[il, ic] = dummy
                        ic += 1
                ## the equilibrium value of the non conserved moments
                ## does not depend on the non conserved moments
                #for n_j in xrange(len(self.ind_noncons)):
                #    for k_j in self.ind_noncons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                il += 1

        S = sp.zeros(nvtot, nvtot)
        il = 0
        for n_i in range(ns):
            for k_i in self.ind_cons[n_i]:
                S[il, il] = self.s_symb[n_i][k_i]
                il += 1
        for n_i in range(ns):
            for k_i in self.ind_noncons[n_i]:
                S[il, il] = self.s_symb[n_i][k_i]
                il += 1

        J = sp.eye(nvtot) - S + S * Eeq
        # print(J)

        t1 = mpi.Wtime()
        print("Initialization time: ", t1-t0)

        matA, matB, matC, matD = [], [], [], []
        Dn = sp.zeros(nvtot, nvtot)
        nnn = sp.Symbol('nnn')
        for k in range(nvtot):
            Dnk = (- sum([LA * sp.Integer(v[alpha, k]) * drondx[alpha] for alpha in range(self.dim)]))**nnn/sp.factorial(nnn)
            Dn[k,k] = Dnk
        dummyn = M * Dn * invM * J
        for n in range(order+1):
            dummy = dummyn.subs([(nnn,n)])
            dummy.simplify()
            matA.append(dummy[:N, :N])
            matB.append(dummy[:N, N:])
            matC.append(dummy[N:, :N])
            matD.append(dummy[N:, N:])

        t2 = mpi.Wtime()
        print("Compute A, B, C, D: ", t2-t1)
        # for k in range(len(matA)):
        #     print("k={0}".format(k))
        #     print(matA[k][0,0])
        #     print(matB[k][0,0])
        #     print(matC[k][0,0])
        #     print(matD[k][0,0])

        iS = S[N:,N:].inv()
        matC[0] = iS * matC[0]
        matC[0].simplify()
        # Gamma[0][0] = matA[1]
        #
        # Gamma[1][0] = matA[2]
        # Gamma[1][1] = matA[1] * Gamma[0][0]
        #
        # Gamma[2][0] = matA[3]
        # Gamma[2][1] = matA[1] * Gamma[1][0] + matA[2] * Gamma[0][0]
        # Gamma[2][2] = matA[1] * Gamma[1][1]
        # ...
        Gamma = []
        for k in range(1,order+1):
            for j in range(1,k+1):
                matA[k] += matB[j] * matC[k-j]
            matA[k].simplify()
            Gammak = [None]
            for j in range(1,k):
                Gammakj = sp.zeros(N,N)
                for l in range(1, k-j+1):
                    Gammakj += matA[l] * Gamma[k-l-1][j-1]
                Gammakj.simplify()
                Gammak.append(Gammakj)
            Gamma.append(Gammak)
            for j in range(1,k):
                matA[k] -= Gamma[k-1][j]/sp.factorial(j+1)
            matA[k].simplify()
            Gamma[-1][0] = matA[k].copy()
            for j in range(1,k+1):
                matC[k] += matD[j] * matC[k-j]
            for j in range(k):
                Kkj = sp.zeros(nvtot-N, N)
                for l in range(k-j):
                    Kkj += matC[l] * Gamma[k-l-1][j]
                matC[k] -= Kkj/sp.factorial(j+1)
            matC[k] = iS * matC[k]
            matC[k].simplify()
        t3 = mpi.Wtime()
        print("Compute alpha, beta: ", t3-t2)
        # for k in range(len(matA)):
        #     print("k={0}".format(k))
        #     print(matA[k][0,0])
        #     print(matC[k][0,0])
        # k = 0
        # for Gammak in Gamma:
        #     print("***** k={0} *****".format(k))
        #     for Gammakj in Gammak:
        #         print(sp.simplify(Gammakj[0,0]))
        #     k += 1

        W = sp.zeros(N, 1)
        dummy = [0]
        sp.init_printing()
        for n in range(ns):
            dummy.append(dummy[-1] + len(self.ind_cons[n]))
        for wk, ik in self.consm.items():
            W[dummy[ik[0]] + self.ind_cons[ik[0]].index(ik[1]),0] = wk
        self.consistency = {}
        for k in range(N):
            wk = W[k,0]
            self.consistency[wk] = {'lhs':[sp.simplify(drondt * W[k,0]), sp.simplify(-(matA[1]*W)[k,0])]}
            lhs = sp.simplify(sum(self.consistency[wk]['lhs']))
            dummy = []
            for n in range(1,order):
                dummy.append(sp.simplify(time_step**n * (matA[n+1]*W)[k,0]))
            self.consistency[wk]['rhs'] = dummy
            rhs = sp.simplify(sum(self.consistency[wk]['rhs']))
            print("\n" + "*"*50)
            print("Conservation equation for {0} at order {1}".format(wk, order))
            sp.pprint(lhs)
            print(" "*10, "=")
            sp.pprint(rhs)
            print("*"*50)
        #print self.consistency

        t4 = mpi.Wtime()
        print("Compute equations: ", t4-t3)
        print("Total time: ", t4-t0)


def test_1D(opt):
    dim = 1 # spatial dimension
    la = 1.
    print("\n\nTest number {0:d} in {1:d}D:".format(opt,dim))
    dico = {'dim':dim, 'scheme_velocity':la}
    if (opt == 0):
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':[2,0,1],
           'polynomials':Matrix([1,la*X,X**2/2]),
           'equilibrium':Matrix([u[0][0], u[0][1], (0.5*la)**2/2*u[0][0]]),
           'relaxation_parameters':[0,0,1.9]
           }
    elif (opt == 1):
        dico['number_of_schemes'] = 2 # number of elementary schemes
        dico[0] = {'velocities':[2,1],
           'polynomials':Matrix([1,la*X]),
           'equilibrium':Matrix([u[0][0], u[1][0]]),
           'relaxation_parameters':[0,1.5]
           }
        dico[1] = {'velocities':[2,1],
           'polynomials':Matrix([1,la*X]),
           'equilibrium':Matrix([u[1][0], u[0][0]]),
           'relaxation_parameters':[0,1.2]
           }
    elif (opt == 2):
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':list(range(5)),
           'polynomials':Matrix([1, la*X, X**2/2, X**3/2, X**4/2]),
           'equilibrium':Matrix([u[0][0], u[0][1], (0.5*la)**2/2*u[0][0], 0, 0]),
           'relaxation_parameters':[0,0,1.9, 1., 1.]
           }
    try:
        LBMscheme = Scheme(dico)
        print(LBMscheme)
        return 1
    except:
        return 0

def test_2D(opt):
    dim = 2 # spatial dimension
    la = 1.
    print("\n\nTest number {0:d} in {1:d}D:".format(opt,dim))
    dico = {'dim':dim, 'scheme_velocity':la}
    if (opt == 0):
        dico['number_of_schemes'] = 2 # number of elementary schemes
        dico[0] = {'velocities':list(range(1,5)),
           'polynomials':Matrix([1, la*X, la*Y, X**2-Y**2]),
           'equilibrium':Matrix([u[0][0], .1*u[0][0], 0, 0]),
           'relaxation_parameters':[0, 1, 1, 1]
           }
        dico[1] = {'velocities':list(range(5)),
           'polynomials':Matrix([1, la*X, la*Y, X**2+Y**2, X**2-Y**2]),
           'equilibrium':Matrix([u[1][0], 0, 0, 0.1*u[1][0], 0]),
           'relaxation_parameters':[0, 1, 1, 1, 1]
           }
    elif (opt == 1):
        rhoo = 1.
        dummy = 1./(la**2*rhoo)
        qx2 = dummy*u[0][1]**2
        qy2 = dummy*u[0][2]**2
        q2  = qx2+qy2
        qxy = dummy*u[0][1]*u[0][2]
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':list(range(9)),
           'polynomials':Matrix([1, la*X, la*Y, 3*(X**2+Y**2)-4, (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2, 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y, X**2-Y**2, X*Y]),
           'equilibrium':Matrix([u[0][0], u[0][1], u[0][3], -2*u[0][0] + 3*q2, u[0][0]+1.5*q2, u[0][1]/la, u[0][2]/la, qx2-qy2, qxy]),
           'relaxation_parameters':[0, 0, 0, 1, 1, 1, 1, 1, 1]
           }
    try:
        LBMscheme = Scheme(dico)
        print(LBMscheme)
        return 1
    except:
        return 0

if __name__ == "__main__":
    k = 1
    compt = 0
    while (k==1):
        k = test_1D(compt)
        compt += 1
    k = 1
    compt = 0
    while (k==1):
        k = test_2D(compt)
        compt += 1
