# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Description of a LBM scheme
"""
import sys
import logging
from textwrap import dedent

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, Eq

from .stencil import Stencil
from .validator import validate
from .symbolic import rel_ux, rel_uy, rel_uz, alltogether, SymbolicVector
#pylint: disable=too-many-lines

log = logging.getLogger(__name__) #pylint: disable=invalid-name

def allfactor(M):
    """
    Factorize all the elements of sympy matrix M

    Parameters
    ----------

    M : sympy matrix
       matrix to factorize

    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i, j] = M[i, j].factor()

def param_to_tuple(param):
    """
    Convert param dictionary to a list of keys and a list of values.

    Parameters
    ----------

    param : dictionary
        parameters

    Returns
    -------

    list
        the keys of param
    list
        the values of param

    """

    if param is not None:
        keys, values = list(param.keys()), list(param.values())
    else:
        keys, values = (), ()
    return keys, values

#pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-statements
class Scheme:
    """
    Create the class with all the needed informations for each elementary scheme.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`
      - dim : spatial dimension (optional if the `box` is given)
      - scheme_velocity : the value of the ratio space step over time step
        (la = dx / dt)
      - schemes : a list of dictionaries, one for each scheme

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
    nschemes : int
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

    Examples
    --------

    see demo/examples/scheme/

    """
    def __init__(self,
                 dico,
                 check_inverse=False,
                 need_validation=True):

        if need_validation:
            validate(dico, __class__.__name__) #pylint: disable=undefined-variable

        self.stencil = Stencil(dico, need_validation=False)
        self.dim = self.stencil.dim

        # symbolic parameters
        self.param = dico.get('parameters', {})

        self.la = dico['scheme_velocity']

        # set relative velocity
        self.rel_vel = dico.get('relative_velocity', None)

        # fix the variables of time and space
        self.symb_t, self.symb_coord = self._get_space_and_time_symbolic()

        self.nschemes = self.stencil.nstencils
        scheme = dico['schemes']


        self._check_entry_size(scheme, 'relaxation_parameters')
        self.s = SymbolicVector([r for s in scheme for r in s['relaxation_parameters']])

        # TODO: add the possibility to have vectorial schemes when M matrix is defined
        if len(scheme) == 1 and 'M' in scheme[0]:
            self.M = scheme[0]['M']
            self.invM = self.M.inv()
            self.Tu = sp.eye(*self.M.shape)
            self.Tmu = sp.eye(*self.M.shape)
            self.P = []
        else:
            self._check_entry_size(scheme, 'polynomials')
            self.P = sp.Matrix([p for s in scheme for p in s['polynomials']])
            self.M, self.invM, self.Tu, self.Tmu = self._create_moments_matrices()

        self._source_terms = [s.get('source_terms', None) for s in scheme]
        self.EQ = self._get_equilibrium(scheme)

        self.s_no_swap = self.s.copy()
        self.EQ_no_swap = self.EQ.copy()
        self.M_no_swap = self.M.copy()
        self.invM_no_swap = self.invM.copy()

        self.consm = self._get_conserved_moments(scheme)
        # put conserved moments at the beginning of EQ and s
        # and permute M, invM, Tu, and Tmu accordingly
        self._permute_consm_in_front()
        if self.rel_vel is not None:
            self.Tu_no_swap = self.Tu.copy()

        self._check_inverse(self.M, self.invM, 'M')
        self._check_inverse_of_Tu()

        log.info(self.__str__())

    def _get_space_and_time_symbolic(self):
        symb_t = self.param.get('time', sp.Symbol('t'))

        symb_coord = [None, None, None]
        symb_coord[0] = self.param.get('space_x', sp.Symbol('X'))
        symb_coord[1] = self.param.get('space_y', sp.Symbol('Y'))
        symb_coord[2] = self.param.get('space_z', sp.Symbol('Z'))

        return symb_t, symb_coord

    def _get_equilibrium(self, scheme):
        eq = []
        for i, s in enumerate(scheme):
            feq = s.get('feq', None)
            meq = s.get('equilibrium', None)
            if feq and meq:
                log.error("Error in the creation of the scheme %d: you can have only 'feq' or 'equilibrium'", i)
                sys.exit()
            if meq:
                eq.append(meq)
            if feq:
                sli = slice(self.stencil.nv_ptr[i], self.stencil.nv_ptr[i+1])
                meq_tmp = self.M[sli, sli]*feq[0](self.stencil.get_all_velocities(i), *feq[1])
                meq_tmp.simplify()
                eq.append([e for e in meq_tmp])
        return SymbolicVector([e for sublist in eq for e in sublist])

    def _permute_consm_in_front(self):
        self.permutations = []
        for ic, c in enumerate(self.consm.values()):
            self.permutations.append([ic, c])

        for p in self.permutations:
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

    def _check_entry_size(self, schemes, key):
        for i, s in enumerate(schemes):
            ls = len(s[key])
            nv = self.stencil.nv[i]
            if ls != nv:
                log.error(dedent("""\
                                the size of the entry for the key {0} in the scheme {1}
                                has not the same size of the stencil {1}: {2}, {3}""".format(key, i, ls, nv)))

    def __str__(self):
        from .utils import header_string
        from .jinja_env import env
        template = env.get_template('scheme.tpl')
        P = []
        EQ = []
        s = []
        header_scheme = []
        for k in range(self.nschemes):
            myslice = slice(self.stencil.nv_ptr[k], self.stencil.nv_ptr[k+1])
            header_scheme.append(header_string("Scheme %d"%k))
            P.append(sp.pretty(sp.Matrix(self.P[myslice])))
            EQ.append(sp.pretty(sp.Matrix(self.EQ_no_swap[myslice])))
            s.append(sp.pretty(sp.Matrix(self.s_no_swap[myslice])))

        if self.rel_vel:
            addons = {'rel_vel': self.rel_vel,
                      'Tu': sp.pretty(self.Tu_no_swap)
                     }
        else:
            addons = {}

        return template.render(header=header_string("Scheme information"),
                               scheme=self,
                               consm=sp.pretty(list(self.consm.keys())),
                               header_scheme=header_scheme,
                               P=P,
                               EQ=EQ,
                               s=s,
                               M=sp.pretty(self.M_no_swap),
                               invM=sp.pretty(self.invM_no_swap),
                               **addons
                              )

    def __repr__(self):
        return self.__str__()

    def vue(self):
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        import base64
        from traitlets import Unicode

        try:
            import ipyvuetify as v
            import ipywidgets as widgets
        except ImportError:
            raise ImportError("Please install ipyvuetify")

        panels = []
        plt.ioff()
        for k in range(self.nschemes):
            myslice = slice(self.stencil.nv_ptr[k], self.stencil.nv_ptr[k+1])
            P = [sp.latex(p, mode='equation*') for p in self.P[myslice]]
            EQ = [sp.latex(eq, mode='equation*') for eq in self.EQ_no_swap[myslice]]
            s = [sp.latex(s, mode='equation*') for s in self.s_no_swap[myslice]]

            view = self.stencil.visualize(k=k)
            view.fig.canvas.header_visible = False

            panels.append(
                v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=[f'Scheme {k}'], class_='title'),
                    v.ExpansionPanelContent(children=[
                        v.Card(children=[
                            v.CardTitle(style_='border-bottom: 1px solid black;', children=['Velocities']),
                            v.CardText(children=[v.Row(children=[view.fig.canvas], justify='center')]),
                        ], class_="ma-2", elevation=5),
                        v.Card(children=[
                            v.CardTitle(style_='border-bottom: 1px solid black;', children=['Polynomials']),
                            v.CardText(children=[v.Row(children=[widgets.HTMLMath(p)], justify='center') for p in P])
                        ], class_="ma-2", elevation=5),
                        v.Card(children=[
                            v.CardTitle(style_='border-bottom: 1px solid black;', children=['Equilibria']),
                            v.CardText(children=[v.Row(children=[widgets.HTMLMath(eq)], justify='center') for eq in EQ])
                        ], class_="ma-2", elevation=5),
                        v.Card(children=[
                            v.CardTitle(style_='border-bottom: 1px solid black;', children=['Relaxation parameters']),
                            v.CardText(children=[v.Row(children=[widgets.HTMLMath(s_i)], justify='center') for s_i in s])
                        ], class_="ma-2", elevation=5),
                    ])
                ], class_='ma-2 pa-2')
            )

        plt.ion()

        panels.append(
            v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=['Moments matrix'], class_='title'),
                    v.ExpansionPanelContent(children=[v.Row(children=[widgets.HTMLMath(f"{sp.latex(self.M_no_swap, mode='equation*')}")], justify='center')])
            ], class_='ma-2 pa-2')
        )

        panels.append(
            v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=['Inverse of moments matrix'], class_='title'),
                    v.ExpansionPanelContent(children=[v.Row(children=[widgets.HTMLMath(f"{sp.latex(self.invM_no_swap, mode='equation*')}")], justify='center')])
            ], class_='ma-2 pa-2')
        )

        return v.ExpansionPanels(children=panels, multiple=True)

    def _repr_mimebundle_(self, **kwargs):
        data = {
            'text/plain': repr(self),
        }
        data['application/vnd.jupyter.widget-view+json'] = {
            'version_major': 2,
            'version_minor': 0,
            'model_id': self.vue()._model_id
        }
        return data

    def _create_moments_matrices(self):
        """
        Create the moments matrices M and M^{-1} used to transform the repartition functions into the moments

        Three versions of these matrices are computed:

          - a sympy version M and invM for each scheme
          - a numerical version Mnum and invMnum for each scheme
          - a global numerical version MnumGlob and invMnumGlob for all the schemes
        """
        M_, invM_, Mu_, Tu_ = [], [], [], []
        u_tild = sp.Matrix([rel_ux, rel_uy, rel_uz])

        LA = self.la

        compt = 0
        for iv, v in enumerate(self.stencil.v):
            p = self.P[self.stencil.nv_ptr[iv] : self.stencil.nv_ptr[iv+1]]
            compt += 1
            lv = len(v)
            M_.append(sp.zeros(lv, lv))
            Mu_.append(sp.zeros(lv, lv))
            for i in range(lv):
                for j in range(lv):
                    sublist = [(str(self.symb_coord[d]), sp.Integer(v[j].v[d])*LA) for d in range(self.dim)]
                    M_[-1][i, j] = p[i].subs(sublist)

                    if self.rel_vel is not None:
                        sublist = [(str(self.symb_coord[d]), v[j].v[d]*LA - u_tild[d]) for d in range(self.dim)]
                        Mu_[-1][i, j] = p[i].subs(sublist)

            invM_.append(M_[-1].inv())
            Tu_.append(Mu_[-1]*invM_[-1])

        gshape = (self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1])
        Tu = sp.eye(gshape[0])
        M = sp.zeros(*gshape)
        invM = sp.zeros(*gshape)

        try:
            for k in range(self.nschemes):
                nvk = self.stencil.nv[k]
                for i in range(nvk):
                    for j in range(nvk):
                        index = self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j
                        M[index] = M_[k][i, j]
                        invM[index] = invM_[k][i, j]

                        if self.rel_vel is not None:
                            Tu[index] = Tu_[k][i, j]
        except TypeError:
            log.error("Unable to convert to float the expression %s or %s.\nCheck the 'parameters' entry.", M[k][i, j], invM[k][i, j]) #pylint: disable=undefined-loop-variable
            sys.exit()

        alltogether(Tu, nsimplify=True)
        alltogether(M, nsimplify=True)
        alltogether(invM, nsimplify=True)
        Tmu = Tu.subs(list(zip(u_tild, -u_tild)))
        return M, invM, Tu, Tmu

    def _check_inverse_of_Tu(self):
        # verification
        res = self.Tu*self.Tmu
        alltogether(res)
        gshape = self.stencil.nv_ptr[-1]
        test = res == sp.eye(gshape)
        if not test:
            log.warning("The property on the translation matrix is not verified\n T(u) * T(-u) is not identity !!!")

    def _check_inverse(self, M, invM, matrix_name):
        """
        Check if the product of a sympy matrix and its inverse
        is identity.

        Parameters
        ----------

        M : sympy.Matrix
            the matrix

        invM : sympy.Matrix
            the inverse of M

        matrix_name : string
            the name of the matrix

        FIXME: the name of the matrix should be obtain via M.

        """
        # verification
        gshape = self.stencil.nv_ptr[-1]
        dummy = M*invM
        alltogether(dummy)
        test = dummy == sp.eye(gshape)
        if not test:
            log.warning("Problem %s * inv%s is not identity !!!", matrix_name, matrix_name)

    def _get_conserved_moments(self, scheme):
        """
        return conserved moments and their indices in the scheme entry.

        Parameters
        ----------

        scheme : dictionnary
            description of the LBM schemes

        Returns
        -------

        dictionary
            the keys are the conserved moments and
            the values their indices in the LBM schemes.

        """
        from collections import OrderedDict
        consm_tmp = [s.get('conserved_moments', None) for s in scheme]
        consm = OrderedDict()

        for i in range(len(self.stencil.nv_ptr)-1):
            leq = self.EQ[self.stencil.nv_ptr[i]:self.stencil.nv_ptr[i+1]]
            cm_ieq = consm_tmp[i]
            if cm_ieq is not None:
                if isinstance(cm_ieq, (sp.Symbol, sp.IndexedBase)):
                    consm[cm_ieq] = self.stencil.nv_ptr[i] + leq.index(cm_ieq)
                else:
                    for c in cm_ieq:
                        consm[c] = self.stencil.nv_ptr[i] + leq.index(c)
        return consm

    def _get_indices_cons_noncons(self):
        """
        return the list of the conserved moments and the list of the non conserved moments

        Returns
        -------

        list
            the indices of the conserved moments
        list
            the indices of the non conserver moments

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

    def set_source_terms(self, scheme):
        """
        set the source terms functions for the conserved moments.

        Parameters
        ----------

        scheme : dictionnary
            description of the LBM schemes

        Returns
        -------

        dictionnary
            the keys are the indices of the conserved moments
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
                        log.error(dedent("""\
                                         Error in the creation of the scheme: wrong dictionnary
                                         the key `source_terms` should contain a dictionnary with
                                            key: the moment concerned
                                                 should be the name of the moment as a string or
                                                 a sympy Symbol or an integer
                                            value: the value of the source term
                                                 should be a float or a sympy expression
                                         """))
                        sys.exit()
                    source_terms[-1][indices[1]] = v
                    is_empty = False
        if is_empty:
            return None
        else:
            return source_terms
