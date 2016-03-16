# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
import sympy as sp
import re
from six.moves import range

from .base import Generator, INDENT
from .utils import matMult, load_or_store, import_mathematical_function
from .ode_schemes import *

class NumpyGenerator(Generator):
    """
    the default generator of code,
    subclass of :py:class:`Generator<pyLBM.generator.Generator>`

    Parameters
    ----------

    build_dir : string, optional
      the directory where the code has to be written

    Notes
    -----

    This generator can always be used but is not the most efficient.

    Attributes
    ----------

    build_dir : string
      the directory where the code is written
    f : file identifier
      the file where the code is written
    code : string
      the generated code

    Methods
    -------

    transport :
      generate the code of the transport phase
    relaxation :
      generate the code of the relaxation phase
    equilibrium :
      generate the code of the equilibrium
    m2f :
      generate the code to compute the distribution functions
      from the moments
    f2m :
      generate the code to compute the moments
      from the distribution functions
    """
    def __init__(self, build_dir=None):
        Generator.__init__(self, build_dir)
        self.sameF = True

    def setup(self):
        """
        initialization of the .py file to use numpy
        """
        self.code += import_mathematical_function('Numpy')

    def transport(self, ns, stencil, dtype = 'f8'):
        """
        generate the code of the transport phase

        Parameters
        ----------

        ns : int
          the number of elementary schemes
        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the transport phase in the attribute ``code``.
        """
        self.code += "def transport(f):\n"
        v = stencil.get_all_velocities()
        code, is_empty = load_or_store('f', 'f', -v, v, self.sorder, indent=INDENT)
        if is_empty:
            self.code += INDENT + "pass\n"
        else:
            self.code += code
        self.code += "\n"

    def equilibrium(self, ns, stencil, eq, dtype = 'f8'):
        """
        generate the code of the projection on the equilibrium

        Parameters
        ----------

        ns : int
          the number of elementary schemes
        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities
        eq : sympy matrix
          the equilibrium (formally given)
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the equilibrium function in the attribute ``code``.
        """
        self.code += "def equilibrium(m):\n"

        def sub(g):
            slices = [':']*len(self.sorder)
            i = int(g.group('i'))
            j = int(g.group('j'))
            slices[self.sorder[0]] = str(stencil.nv_ptr[i] + j)
            return '[%s]'%(', '.join(slices))

        slices = [':']*len(self.sorder)

        dummy_test = True
        for k in range(ns):
            for i in range(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    slices[self.sorder[0]] = str(stencil.nv_ptr[k] + i)
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i]))
                        self.code += INDENT + "m[%s] = %s\n"%(', '.join(slices), res)
                    else:
                        self.code += INDENT + "m[%s] = 0.\n"%(', '.join(slices))
                    dummy_test = False
        if dummy_test:
            self.code += INDENT + "pass\n"
        self.code += "\n"

    def relaxation(self, ns, stencil, s, eq, dicoST = None, dtype = 'f8'):
        """
        generate the code of the relaxation phase

        Parameters
        ----------

        ns : int
          the number of elementary schemes
        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities
        s : list of list of double
          the values of the relaxation parameters
        eq : sympy matrix
          the equilibrium (formally given)
        dicoST : dictionary
          the dictionary for the source term
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the relaxation phase in the attribute ``code``.
        """
        self.code += "def relaxation(m, tn=0., k=0., x=0, y=0, z=0):\n"

        def sub(g):
            slices = [':']*len(self.sorder)
            i = int(g.group('i'))
            j = int(g.group('j'))
            slices[self.sorder[0]] = str(stencil.nv_ptr[i] + j)
            return '[%s]'%(', '.join(slices))

        slices = [':']*len(self.sorder)

        dummy_test = True
        # half of the source term code
        test_source_term = False
        if dicoST is not None:
            st = dicoST['ST']
            vart = dicoST['vart']
            ode_solver = dicoST['ode_solver']
            indices_m = []
            f = []
            for k in range(ns):
                for i in range(stencil.nv[k]):
                    if st[k][i] is not None and st[k][i] != 0:
                        test_source_term = True
                        indices_m.append((k, i))
                        f.append(str(st[k][i]))
            if test_source_term:
                dummy_test = False
                ode_solver.parameters(indices_m, f, vart, dt='0.5*k', indent=INDENT, add_copy = ".copy()")
                code_source_term = ode_solver.cpt_code()
                code_source_term = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub, code_source_term)
                code_source_term = re.sub(str(vart), 'tn', code_source_term)

        # relaxation code
        code_relaxation = ''
        for k in range(ns):
            for i in range(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    slices[self.sorder[0]] = str(stencil.nv_ptr[k] + i)
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i]))
                        code_relaxation += INDENT + "m[{0}] += {1:.16f}*({2} - m[{0}])\n".format(', '.join(slices), s[k][i], res)
                    else:
                        code_relaxation += INDENT + "m[{0}] *= (1. - {1:.16f})\n".format(', '.join(slices), s[k][i])
                    dummy_test = False

        if test_source_term:
            N = ode_solver.nb_of_floors
            if N>0:
                self.code += INDENT + 'import numpy as np\n'
                for k in range(N):
                    self.code += INDENT + 'dummy{1:1d} = [np.zeros(m[0][0].shape)]*{0}\n'.format(len(indices_m), k)
            self.code += code_source_term
            self.code += code_relaxation
            self.code += code_source_term
        else:
            self.code += code_relaxation
        if dummy_test:
            self.code += INDENT + "pass\n"
        self.code += "\n"

    def m2f(self, A, dim, dtype = 'f8'):
        """
        generate the code to compute the distribution functions from the moments

        Parameters
        ----------

        A : numpy array
          the matrix M^(-1) in the d'Humieres framework
        k : int
          the number of the stencil for which the code has to be generated
        dim : int
          the space dimension (unused)
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the m2f function in the attribute ``code``.

        Notes
        -----

        For the NumpyGenerator, it seems to be more efficient
        to generate one function per elementary scheme
        because the index corresponding to the stencil is the first index.
        """
        self.code += "def m2f(m, f):\n"
        self.code += matMult(A, 'm', 'f', self.sorder, indent=INDENT)
        self.code += "\n"

    def f2m(self, A, dim, dtype = 'f8'):
        """
        generate the code to compute the moments from the distribution functions

        Parameters
        ----------

        A : numpy array
          the matrix M in the d'Humieres framework
        k : int
          the number of the stencil for which the code has to be generated
        dim : int
          the space dimension (unused)
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the f2m function in the attribute ``code``.

        Notes
        -----

        For the NumpyGenerator, it seems to be more efficient
        to generate one function per elementary scheme
        because the index corresponding to the stencil is the first index.
        """
        #self.code += "def f2m_{0:d}(f, m):\n".format(k)
        self.code += "def f2m(f, m):\n"
        self.code += matMult(A, 'f', 'm', self.sorder, indent=' '*4)
        self.code += "\n"

    def onetimestep(self, stencil):
        self.code += """
def onetimestep(m, f, fnew, in_or_out, valin, tn=0., dt=0., x=0, y=0, z=0):
    transport(f)
    f2m(f, m)
    relaxation(m, tn, dt, x, y, z)
    m2f(m, f)

"""
