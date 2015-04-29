# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
import numpy as np
import sympy as sp
import re

from .base import Generator
from .utils import matMult, load_or_store

class CythonGenerator(Generator):
    """
    generate the code by using `Cython <http://cython.org>`_,
    subclass of :py:class:`Generator<pyLBM.generator.Generator>`

    Parameters
    ----------

    build_dir : string, optional
      the directory where the code has to be written

    Notes
    -----

    This generator can be used if cython is installed and is more
    efficient than :py:class:`NumpyGenerator <pyLBM.generator.NumpyGenerator>`

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

    setup :
      setup function to import the module numba
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
    onetimestep :
      generate the code to compute one time step of the
      Lattice Boltzmann method
    compile :
      compile the cython code
    """
    def __init__(self, build_dir=None):
        Generator.__init__(self, build_dir, suffix='.pyx')

    def setup(self):
        """
        initialization of the .pyx file to use cython
        """
        self.code += """
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#import cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
"""

    def m2f(self, A, k, dim, dtype = 'double'):
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

        For the CythonGenerator, it seems to be more efficient
        to generate one function for the global scheme
        because the index corresponding to the stencil is the last index.
        """
        self.code += "cdef void m2f_loc({0} *m, {0} *f) nogil:\n".format(dtype)

        self.code += matMult(A, 'm', 'f', '\t')
        self.code += "\n"

        self.code += "def m2f({0}[:, ::1] m, {0}[:, ::1] f):\n".format(dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[0] \n"
        self.code += "\tfor i in xrange(n):\n"

        self.code += matMult(A, 'm', 'f', '\t'*2, prefix='i, ')
        self.code += "\n"

    def f2m(self, A, k, dim, dtype = 'double'):
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

        For the CythonGenerator, it seems to be more efficient
        to generate one function for the global scheme
        because the index corresponding to the stencil is the last index.
        """
        self.code += "cdef void f2m_loc({0} *f, {0} *m) nogil:\n".format(dtype)

        self.code += matMult(A, 'f', 'm', '\t')
        self.code += "\n"

        self.code += "def f2m({0}[:, ::1] f, {0}[:, ::1] m):\n".format(dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[0] \n"
        self.code += "\tfor i in xrange(n):\n"

        self.code += matMult(A, 'f', 'm', '\t'*2, prefix='i, ')
        self.code += "\n"

    def transport(self, ns, stencil, dtype = 'double'):
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
        for i in xrange(stencil.dim):
            stmp += ', int i{0}'.format(i)

        get_f = "cdef void get_f({0} *floc, {0}[:{1}:1] f{2}) nogil:\n".format(dtype, ', :'*stencil.dim, stmp)
        set_f = "cdef void set_f({0} *floc, {0}[:{1}:1] f{2}) nogil:\n".format(dtype, ', :'*stencil.dim, stmp)

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
        self.code += "def equilibrium(double[:, ::1] m):\n"
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[0] \n"
        self.code += "\tfor i in xrange(n):\n"

        def subpow(g):
            s = g.group('m')
            for i in xrange(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            return s

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[i, ' + str(stencil.nv_ptr[i] + j) + ']'


        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        str2input = str(eq[k][i])
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", subpow, str2input)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub, res)

                        self.code += "\t\tm[i, %d] = %s\n"%(stencil.nv_ptr[k] + i, res)
                    else:
                        self.code += "\t\tm[i, %d] = 0.\n"%(stencil.nv_ptr[k] + i)
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
        self.code += "cdef void relaxation(double *m) nogil:\n"

        def subpow(g):
            s = g.group('m')
            for i in xrange(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            return s

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ']'

        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        str2input = str(eq[k][i])
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", subpow, str2input)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub, res)

                        self.code += "\t\tm[{0:d}] += {1:.16f}*({2} - m[{0:d}])\n".format(stencil.nv_ptr[k] + i, s[k][i], res)
                    else:
                        self.code += "\t\tm[{0:d}] *= (1. - {1:.16f})\n".format(stencil.nv_ptr[k] + i, s[k][i])
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
def onetimestep(double[{0}::1] m, double[{0}::1] f, double[{0}::1] fnew, double[{2}]in_or_out, double valin):
\tcdef:
\t\tdouble floc[{1}]
\t\tdouble mloc[{1}]
""".format(s, stencil.nv_ptr[-1], ext)

        s1 = ''
        for i in xrange(stencil.dim):
            s1 += "\t\tint i{0}, n{0} = f.shape[{0}]\n".format(i)
        self.code += s1
        self.code += "\t\tint i, nv = f.shape[{0}]\n".format(stencil.dim)
        self.code += "\t\tint nvsize = {0}\n".format(stencil.nv_ptr[-1])
        tab = '\t'
        s1 = ''
        ext = ''

        for i in xrange(stencil.dim):
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
        bld = open(self.f.name.replace('.pyx', '.pyxbld'), "w")
        code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     #extra_compile_args = ['-O3', '-fopenmp, '-w'],
                     #extra_link_args= ['-fopenmp'])
                     extra_compile_args = ['-O3', '-w']
                     #extra_compile_args = ['-O3', '-fopenmp', '-w'],
                     #extra_link_args= ['-fopenmp'])
                    )
        """
        bld.write(code)
        bld.close()

        import pyximport
        pyximport.install(build_dir= self.build_dir, inplace=True)
