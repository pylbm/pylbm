# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
import numpy as np
import sympy as sp
import re
from six.moves import range

from .base import Generator
from .utils import matMult, load_or_store

class PythranGenerator(Generator):
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

    def setup(self):
        """
        initialization of the .pyx file to use cython
        """
        self.code += "import numpy as np\n"

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
        stmp = ''
        for i in range(stencil.dim):
            stmp += ', i{0}'.format(i)

        get_f = "def get_f(floc, f{0}):\n".format(stmp)
        set_f = "def set_f(floc, f{0}):\n".format(stmp)

        v = stencil.get_all_velocities()
        get_f += load_or_store('floc', 'f', v, None, vec_form=False, nv_on_beg=False, indent='\t')
        set_f += load_or_store('f', 'floc', None, np.zeros(v.shape), vec_form=False, nv_on_beg=False, indent='\t')

        self.code += get_f + "\n"
        self.code += set_f + "\n"

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

        self.code += "#pythran export equilibrium(float64[][])\n"
        self.code += "def equilibrium(m):\n"

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[:, ' + str(stencil.nv_ptr[i] + j) + ']'

        pref = ':'
        for k in range(ns):
            for i in range(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i]))
                        self.code += "\tm[{0}, {1}] = {2}\n".format(pref, stencil.nv_ptr[k] + i, res)
                    else:
                        self.code += "\tm[{0}, {1}] = 0.\n".format(pref, stencil.nv_ptr[k] + i)
        self.code += "\n"

    def relaxation(self, ns, stencil, s, eq, dtype = 'f8'):
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
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the relaxation phase in the attribute ``code``.
        """
        self.code += "def relaxation(m):\n"

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ']'

        for k in range(ns):
            for i in range(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i]))
                        self.code += "\tm[{0:d}] += {1:.16f}*({2} - m[{0:d}])\n".format(stencil.nv_ptr[k] + i, s[k][i], res)
                    else:
                        self.code += "\tm[{0:d}] *= {1:.16f}\n".format(stencil.nv_ptr[k] + i, 1. - s[k][i])
        self.code += "\n"

    def m2f(self, A, k, dim, dtype = 'f8'):
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
        self.code += "def m2f_loc(m, f):\n"

        self.code += matMult(A, 'm', 'f', '\t')
        self.code += "\n"

        self.code += "#pythran export m2f(float64[][], float64[][])\n"
        self.code += "def m2f(m, f):\n"
        self.code += matMult(A, 'm', 'f', '\t', prefix=':, ')
        self.code += "\n"

    def f2m(self, A, k, dim, dtype = 'f8'):
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
        self.code += "def f2m_loc(f, m):\n"

        self.code += matMult(A, 'f', 'm', '\t')
        self.code += "\n"

        self.code += "#pythran export f2m(float64[][], float64[][])\n"
        self.code += "def f2m(f, m):\n"
        self.code += matMult(A, 'f', 'm', '\t', prefix=':, ')
        self.code += "\n"

    def onetimestep(self, stencil):
        """
        generate the code for one time step of the Lattice Boltzmann method

        Parameters
        ----------

        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities

        Returns
        -------
        code : string
          add the onetimestep function in the attribute ``code``.

        """
        s = ':, '*stencil.dim
        ext = s[:-2]

        self.code += """
#pythran export onetimestep(float64[]{0}, float64[]{0}, float64[]{0}, float64{0}, int32)
def onetimestep(m, f, fnew, in_or_out, valin):
""".format('[]'*stencil.dim)

        s1 = ''
        for i in range(stencil.dim):
            s1 += "\tn{0} = f.shape[{0}]\n".format(i)
        self.code += s1

        self.code += "\tfloc = np.empty({0})\n".format(stencil.nv_ptr[-1])
        self.code += "\tmloc = np.empty({0})\n".format(stencil.nv_ptr[-1])

        tab = '\t'
        s1 = ''
        ext = ''
        for i in range(stencil.dim):
            s1 += tab + 'for i{0} in xrange(n{0}):\n'.format(i)
            ext += 'i{0},'.format(i)
            tab += '\t'
        self.code += s1
        ext = ext[:-1]

        self.code += tab + "if in_or_out[{0}] == valin:\n".format(ext)
        self.code += tab + "\tget_f(floc, f, {0})\n".format(ext)
        self.code += tab + "\tf2m_loc(floc, mloc)\n"
        self.code += tab + "\trelaxation(mloc)\n"
        self.code += tab + "\tm2f_loc(mloc, floc)\n"
        self.code += tab + "\tset_f(floc, fnew, {0})\n".format(ext)

        self.code += "\n"

    def compile(self):
        """
        compile the cython code by using the module
        `pyximport <http://docs.cython.org/src/reference/compilation.html>`_

        Notes
        -----

        The default options of the compilation are -O3 and -w.
        If the compilation can use the option -fopenmp, add it here.
        """
        Generator.compile(self)
        import pythran
        pythran.compile_pythranfile(self.f.name)
