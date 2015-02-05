# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import tempfile
import atexit
import sys
import os
import re
import sympy as sp

from .logs import __setLogger
log = __setLogger(__name__)


def matMult(A, x, y, indent='', prefix='', suffix=''):
    """
    return a string representing the unroll matrix vector operation y = Ax

    Parameters
    ----------
    A : numpy array
      the matrix coefficients
    x : string
      input vector in a string format
    y : string
      output vector in a string format where the matrix vector product is stored
    indent : string
      representing spaces or tabs which are set at the beginning of each line
    prefix : string
      some string to add before indices
    suffix : string
      some string to add after indices

    Returns
    -------
    code : string
      the string representing the unroll matrix vector operation y = Ax

    Examples
    --------

    >>> import numpy as np
    >>> A = np.arange(12).reshape(4,3)
    >>> A[2, :] *= -1

    >>> print matMult(A, 'm', 'f')
    f[0] =  + m[1] + 2.0000000000000000*m[2]
    f[1] =  + 3.0000000000000000*m[0] + 4.0000000000000000*m[1] + 5.0000000000000000*m[2]
    f[2] =  - 6.0000000000000000*m[0] - 7.0000000000000000*m[1] - 8.0000000000000000*m[2]
    f[3] =  + 9.0000000000000000*m[0] + 10.0000000000000000*m[1] + 11.0000000000000000*m[2]

    >>> print matMult(A, 'm', 'f', '\t', prefix = 'i, j, ')
        f[i, j, 0] =  + m[i, j, 1] + 2.0000000000000000*m[i, j, 2]
        f[i, j, 1] =  + 3.0000000000000000*m[i, j, 0] + 4.0000000000000000*m[i, j, 1] + 5.0000000000000000*m[i, j, 2]
        f[i, j, 2] =  - 6.0000000000000000*m[i, j, 0] - 7.0000000000000000*m[i, j, 1] - 8.0000000000000000*m[i, j, 2]
        f[i, j, 3] =  + 9.0000000000000000*m[i, j, 0] + 10.0000000000000000*m[i, j, 1] + 11.0000000000000000*m[i, j, 2]

    >>> print matMult(A, 'm', 'f', '\t', suffix = ', i, j')
        f[0, i, j] =  + m[1, i, j] + 2.0000000000000000*m[2, i, j]
        f[1, i, j] =  + 3.0000000000000000*m[0, i, j] + 4.0000000000000000*m[1, i, j] + 5.0000000000000000*m[2, i, j]
        f[2, i, j] =  - 6.0000000000000000*m[0, i, j] - 7.0000000000000000*m[1, i, j] - 8.0000000000000000*m[2, i, j]
        f[3, i, j] =  + 9.0000000000000000*m[0, i, j] + 10.0000000000000000*m[1, i, j] + 11.0000000000000000*m[2, i, j]

    """
    nvk1, nvk2 = A.shape
    code = ''

    for i in xrange(nvk1):
        code += indent + "{1}[{2}{0:d}{3}] = ".format(i, y, prefix, suffix)
        for j in xrange(nvk2):
            coef = A[i, j]
            scoef = '' if  abs(coef) == 1 else '{0:.16f}*'.format(abs(coef))
            sign = ' + ' if coef > 0 else ' - '

            if coef != 0:
                code += "{1}{2}{3}[{4}{0:d}{5}]".format(j, sign, scoef, x, prefix, suffix)

        code += "\n"

    return code

class Generator:
    """
    the generic class to generate the code

    Parameters
    ----------

    build_dir : string, optional
      the directory where the code is written
    suffix : string, optional
      the suffix of the file where the code is written

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
      default setup function (empty)
    f2m :
      default f2m function (empty)
    m2f :
      default m2f function (empty)
    transport :
      default transport function (empty)
    relaxation :
      default relaxation function (empty)
    onetimestep :
      defulat one time step function (empty)
    compile :
      default compile function (writte the code in the file)
    get_module :
      get the name of the file where the code is written
    exit :
      exit function that erases the code

    Notes
    -----

    With pyLBM, the code can be generated in several langages.
    Each phase of the Lattice Boltzmann Method
    (as transport, relaxation, f2m, m2f, ...) is treated by an optimzed
    function written, compiled, and executed by the generator.

    The generated code can be read by typesetting the attribute
    ``code``.
    """
    def __init__(self, build_dir=None, suffix='.py'):
        self.build_dir = build_dir
        if build_dir is None:
            self.build_dir = tempfile.mkdtemp(suffix='LBM') + '/'
        self.f = tempfile.NamedTemporaryFile(suffix=suffix, prefix=self.build_dir + 'LBM', delete=False)
        sys.path.append(self.build_dir)
        self.code = ''

        atexit.register(self.exit)

        log.info("Temporary file use for code generator :\n{0}".format(self.f.name))
        #print self.f.name

    def setup(self):
        pass

    def f2m(self):
        pass

    def m2f(self,):
        pass

    def transport(self,):
        pass

    def relaxation(self,):
        pass

    def onetimestep(self,):
        pass

    def compile(self):
        log.info("*"*30 + "\n" + self.code + "\n" + "*"*30)
        self.f.write(self.code)
        self.f.close()

    def get_module(self):
        return self.f.name.replace(self.build_dir, "").split('.')[0]

    def exit(self):
        log.info("delete generator")
        #print "delete generator"
        os.unlink(self.f.name)

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
        sys.path.append(self.build_dir)

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
        ind = 0
        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                s1 = ''
                s2 = ''
                toInput = False
                v = stencil.v[k][i].v
                for iv in xrange(len(v)-1, -1, -1):
                    if v[iv] > 0:
                        toInput = True
                        s1 += ', {0:d}:'.format(v[iv])
                        s2 += ', :{0:d}'.format(-v[iv])
                    elif v[iv] < 0:
                        toInput = True
                        s1 += ', :{0:d}'.format(v[iv])
                        s2 += ', {0:d}:'.format(-v[iv])
                    else:
                        s1 += ', :'
                        s2 += ', :'
                if toInput:
                    self.code += "\tf[{0:d}{1}] = f[{0:d}{2}]\n".format(ind, s1, s2)
                ind += 1
        self.code += "\n"

    def equilibrium(self, ns, stencil, eq, la, dtype = 'f8'):
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
        la : double
          the value of the scheme velocity (dx/dt)
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the equilibrium function in the attribute ``code``.
        """
        self.code += "def equilibrium(m):\n"

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ']'

        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i].subs([(sp.symbols('LA'), la),])))
                        self.code += "\tm[%d] = %s\n"%(stencil.nv_ptr[k] + i, res)
                    else:
                        self.code += "\tm[%d] = 0.\n"%(stencil.nv_ptr[k] + i)
        self.code += "\n"

    def relaxation(self, ns, stencil, s, eq, la, dtype = 'f8'):
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
        la : double
          the value of the scheme velocity (dx/dt)
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

        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                                   str(eq[k][i].subs([(sp.symbols('LA'), la),])))
                        self.code += "\tm[{0:d}] += {1:.16f}*({2} - m[{0:d}])\n".format(stencil.nv_ptr[k] + i, s[k][i], res)
                    else:
                        self.code += "\tm[{0:d}] *= (1. - {1:.16f})\n".format(stencil.nv_ptr[k] + i, s[k][i])
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
        self.code += "def m2f_{0:d}(m, f):\n".format(k)
        self.code += matMult(A, 'm', 'f', '\t')
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
        self.code += "def f2m_{0:d}(f, m):\n".format(k)
        self.code += matMult(A, 'f', 'm', '\t')
        self.code += "\n"


class NumbaGenerator(Generator):
    """
    generate the code by using `Numba <http://numba.pydata.org>`_,
    subclass of :py:class:`Generator<pyLBM.generator.Generator>`

    Notes
    -----

    Not yet completely implemented

    Parameters
    ----------

    build_dir : string, optional
      the directory where the code has to be written

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
        self.code += "import numba\n"

    def m2f(self, A, k, dim, dtype = 'f8'):
        self.code += "@numba.jit('void({0}[:, :], {0}[:, :])', nopython=True)\n".format(dtype)
        self.code += "def m2f_{0:d}(m, f):\n".format(k)

        self.code += "\tn = f.shape[1] \n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t"*2
        ext = ', i'
        self.code += matMult(A, 'm', 'f', tab, suffix=ext)
        self.code += "\n"

    def f2m(self, A, k, dim, dtype = 'f8'):
        self.code += "@numba.jit('void({0}[:, :], {0}[:, :])', nopython=True)\n".format(dtype)
        self.code += "def f2m_{0:d}(f, m):\n".format(k)

        self.code += "\tn = m.shape[1] \n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t"*2
        ext = ', i'
        self.code += matMult(A, 'f', 'm', tab, suffix=ext)
        self.code += "\n"

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
        ext = ''
        for i in xrange(dim):
            ext += ', :'

        self.code += "cdef void m2f_loc({0} *m, {0} *f) nogil:\n".format(dtype)

        tab = "\t"
        ext = ''

        self.code += matMult(A, 'm', 'f', tab)
        self.code += "\n"

        self.code += "def m2f({0}[:, ::1] m, {0}[:, ::1] f):\n".format(dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[0] \n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t"*2
        ext = ''
        pref = 'i, '

        self.code += matMult(A, 'm', 'f', tab, suffix=ext, prefix=pref)
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
        ext = ''
        for i in xrange(dim):
            ext += ', :'

        self.code += "cdef void f2m_loc({0} *f, {0} *m) nogil:\n".format(dtype)

        tab = "\t"
        ext = ''

        self.code += matMult(A, 'f', 'm', tab)
        self.code += "\n"

        self.code += "def f2m({0}[:, ::1] f, {0}[:, ::1] m):\n".format(dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[0] \n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t"*2
        ext = ''
        pref = 'i, '

        self.code += matMult(A, 'f', 'm', tab, suffix=ext, prefix=pref)
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

        ind = 0
        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                s = ''
                toInput = False
                v = stencil.v[k][i].v
                lv = len(v)
                for iv in xrange(len(v)-1, -1, -1):
                    if v[iv] > 0:
                        toInput = True
                        s += 'i{0:d} - {1:d}, '.format(lv - 1 - iv, v[iv])
                    elif v[iv] < 0:
                        toInput = True
                        s += 'i{0:d} + {1:d}, '.format(lv - 1 - iv, -v[iv])
                    else:
                        s += 'i{0:d}, '.format(lv - 1 - iv)

                get_f += '\tfloc[{0}] = f[{1}{0}] \n'.format(ind, s)

                s = ''
                for iv in xrange(len(v)-1, -1, -1):
                    s += 'i{0:d}, '.format(lv - 1 - iv)
                set_f += '\tf[{1}{0}] = floc[{0}] \n'.format(ind, s)

                ind += 1


        self.code += get_f + "\n"
        self.code += set_f + "\n"

    def equilibrium(self, ns, stencil, eq, la, dtype = 'f8'):
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
        la : double
          the value of the scheme velocity (dx/dt)
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
                        str2input = str(eq[k][i].subs([(sp.symbols('LA'), la),]))
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", subpow, str2input)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub, res)

                        #res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                        #           str(eq[k][i].subs([(sp.symbols('LA'), la),])))
                        self.code += "\t\tm[i, %d] = %s\n"%(stencil.nv_ptr[k] + i, res)
                    else:
                        self.code += "\t\tm[i, %d] = 0.\n"%(stencil.nv_ptr[k] + i)
        self.code += "\n"

    def relaxation(self, ns, stencil, s, eq, la, dtype = 'f8'):
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
        la : double
          the value of the scheme velocity (dx/dt)
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
                        str2input = str(eq[k][i].subs([(sp.symbols('LA'), la),]))
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
\t\tdouble *floc
\t\tdouble *mloc
""".format(s, stencil.nv_ptr[-1], ext)

        s1 = ''
        for i in xrange(stencil.dim):
            s1 += "\t\tint i{0}, n{0} = f.shape[{0}]\n".format(i)
        self.code += s1
        self.code += "\t\tint i, nv = f.shape[{0}]\n".format(stencil.dim)
        self.code += "\t\tint nvsize = {0}\n".format(stencil.nv_ptr[-1])
        self.code += "\twith nogil, parallel():\n"
        tab = '\t'*2
        s1 = ''
        ext = ''

        self.code += tab + "floc = <double *> malloc(nvsize*sizeof(double))\n"
        self.code += tab + "mloc = <double *> malloc(nvsize*sizeof(double))\n"

        for i in xrange(stencil.dim):
            #s1 += tab + 'for i{0} in xrange(n{0}):\n'.format(i)
            s1 += tab + 'for i{0} in prange(n{0}, schedule="dynamic"):\n'.format(i)
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

        self.code += '\t\t' + "free(floc)\n"
        self.code += '\t\t' + "free(mloc)\n"

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
        if sys.platform == 'darwin':
            code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     extra_compile_args = ['-O3', '-w']
                    )
            """
        else:
            code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     extra_compile_args = ['-O3', '-fopenmp', '-w'],
                     extra_link_args= ['-fopenmp'])
            """
        bld.write(code)
        bld.close()

        import pyximport
        pyximport.install(build_dir= self.build_dir, inplace=True)


if __name__ == "__main__":
    import numpy as np

    c = NumpyGenerator()
    c.setup()

    A = np.ones((9,9))
    c.m2f(A, 0, 3)
    c.f2m(A, 0, 3)
    print c.code

    c.compile()
    exec "from %s import *"%c.get_module()
    print help(m2f_0)
