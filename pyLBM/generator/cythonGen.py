# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
import numpy as np
import sympy as sp
import re
from six.moves import range
from six import string_types
import mpi4py.MPI as mpi

from .base import Generator, INDENT
from .utils import matMult, load_or_store, list_of_cython_functions, dictionnary_of_translation_cython
from .ode_schemes import *

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
        self.sameF = False

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
        self.code += "from libc.math cimport "
        self.code += ", ".join(list_of_cython_functions)
        self.code += "\n\n"

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

        self.code += matMult(A, 'm', 'f', vectorized=False, indent=INDENT)
        self.code += "\n"

        self.code += "def m2f({0}[{1}:1] m, {0}[{1}:1] f):\n".format(dtype, ', '.join([':']*len(self.sorder)))
        self.code += INDENT + "cdef:\n"

        sind = np.argsort(self.sorder[1:])
        for i, x in enumerate(self.sorder[1:]):
            self.code += 2*INDENT + "int i{0}, n{0} = m.shape[{1}] \n".format(i, x)

        indent = INDENT
        for i in sind:
            self.code += indent + "for i{0} in xrange(n{0}):\n".format(i)
            indent += INDENT

        self.code += matMult(A, 'm', 'f', sorder=self.sorder, vectorized=False, indent=indent)
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

        self.code += matMult(A, 'f', 'm', vectorized=False, indent=INDENT)
        self.code += "\n"

        self.code += "def f2m({0}[{1}:1] f, {0}[{1}:1] m):\n".format(dtype, ', '.join([':']*len(self.sorder)))
        self.code += INDENT + "cdef:\n"

        sind = np.argsort(self.sorder[1:])
        for i, x in enumerate(self.sorder[1:]):
            self.code += 2*INDENT + "int i{0}, n{0} = m.shape[{1}] \n".format(i, x)

        indent = INDENT
        for i in sind:
            self.code += indent + "for i{0} in xrange(n{0}):\n".format(i)
            indent += INDENT

        self.code += matMult(A, 'f', 'm', sorder=self.sorder, vectorized=False, indent=indent)

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
        for i in range(stencil.dim):
            stmp += ', int i{0}'.format(i)

        get_f = "cdef void get_f({0} *floc, {0}[:{1}:1] f{2}) nogil:\n".format(dtype, ', :'*stencil.dim, stmp)
        set_f = "cdef void set_f({0} *floc, {0}[:{1}:1] f{2}) nogil:\n".format(dtype, ', :'*stencil.dim, stmp)

        v = stencil.get_all_velocities()
        tmp_get_f, is_empty = load_or_store('floc', 'f', v, None, self.sorder, indent=INDENT, vectorized=False, avoid_copy=False)
        get_f += tmp_get_f
        tmp_set_f, is_empty= load_or_store('f', 'floc', None, np.zeros(v.shape), self.sorder, indent=INDENT, vectorized=False, avoid_copy=False)
        set_f += tmp_set_f

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
        self.code += "def equilibrium(double[{0}:1] m):\n".format(', '.join([':']*len(self.sorder)))
        self.code += INDENT + "cdef:\n"

        sind = np.argsort(self.sorder[1:])
        for i, x in enumerate(self.sorder[1:]):
            self.code += 2*INDENT + "int i{0}, n{0} = m.shape[{1}] \n".format(i, x)

        indent = INDENT
        for i in sind:
            self.code += indent + "for i{0} in xrange(n{0}):\n".format(i)
            indent += INDENT

        def sub_pow(g):
            s = '(' + g.group('m')
            for i in range(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            s += ')'
            return s

        def sub_slices(g):
            i = int(g.group('i'))
            j = int(g.group('j'))
            slices = ['']*len(self.sorder)
            for js, s in enumerate(self.sorder[1:]):
                slices[s] = 'i%d'%js
            slices[self.sorder[0]] = str(stencil.nv_ptr[i] + j)
            return '[%s]'%(', '.join(slices))

        slices = ['']*len(self.sorder)
        for js, s in enumerate(self.sorder[1:]):
            slices[s] = 'i%d'%js

        dummy_test = True
        for k in range(ns):
            for i in range(stencil.nv[k]):
                slices[self.sorder[0]] = str(stencil.nv_ptr[k] + i)
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        str2input = str(eq[k][i])
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", sub_pow, str2input)
                        res = reduce(lambda x, y: x.replace(y, dictionnary_of_translation_cython[y]),
                                     dictionnary_of_translation_cython, res)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub_slices, res)
                        self.code += indent + "m[%s] = %s\n"%(', '.join(slices), res)
                    else:
                        self.code += indent + "m[%s] = 0.\n"%(', '.join(slices))
                    dummy_test = False
        if dummy_test:
            self.code += indent + "pass\n"
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

        def sub_pow(g):
            s = '(' + g.group('m')
            for i in range(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            s += ')'
            return s

        def sub_slices(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ']'

        # relaxation code
        dummy_test = True
        code_relaxation = ''
        for k in range(ns):
            for i in range(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        str2input = str(eq[k][i])
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", sub_pow, str2input)
                        res = reduce(lambda x, y: x.replace(y, dictionnary_of_translation_cython[y]),
                                     dictionnary_of_translation_cython, res)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub_slices, res)
                        code_relaxation += INDENT + "m[{0:d}] += {1:.16f}*({2} - m[{0:d}])\n".format(stencil.nv_ptr[k] + i, s[k][i], res)
                    else:
                        code_relaxation += INDENT + "m[{0:d}] *= (1. - {1:.16f})\n".format(stencil.nv_ptr[k] + i, s[k][i])
                    dummy_test = False

        self.code += code_relaxation
        if dummy_test:
            self.code += INDENT + "pass\n"
        self.code += "\n"


    def source_term(self, ns, stencil, dicoST = None, dtype = 'f8'):
        """
        generate the code of the relaxation phase

        Parameters
        ----------

        ns : int
          the number of elementary schemes
        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities
        dicoST : dictionary
          the dictionary for the source term
        dtype : string, optional
          the type of the data (default 'f8')

        Returns
        -------

        code : string
          add the relaxation phase in the attribute ``code``.
        """
        var_time = sp.Symbol('var_time') # long variable for the time to avoid crazy replacement
        varx, vary, varz = str(dicoST['varx']), str(dicoST['vary']), str(dicoST['varz'])
        self.code += "cdef void source_term(double *m, double tn, double k, double {0}, double {1}, double {2}) nogil:\n".format(varx, vary, varz)

        def sub_pow(g):
            s = '(' + g.group('m')
            for i in range(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            s += ')'
            return s

        def sub_slices(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ']'

        # the source term code
        dummy_test = True
        test_source_term = False
        if dicoST is not None:
            st = dicoST['ST']
            vart = dicoST['vart']
            if isinstance(vart, string_types):
                vart = sp.Symbol(vart)
            ode_solver = dicoST['ode_solver']
            indices_m = []
            f = []
            for k in range(ns):
                for i in range(stencil.nv[k]):
                    if st[k][i] is not None and st[k][i] != 0:
                        test_source_term = True
                        indices_m.append((k, i))
                        # change the name of the time variable
                        if vart is not None:
                            if isinstance(st[k][i], sp.Expr):
                                f.append(str(st[k][i].subs(vart, var_time)))
                            elif isinstance(st[k][i], string_types):
                                f.append(str(st[k][i].replace(str(vart), str(var_time))))
                        else:
                            f.append(str(st[k][i]))
            if test_source_term:
                dummy_test = False
                ode_solver.parameters(indices_m, f, var_time, dt='k', indent=INDENT, add_copy='')
                code_source_term = ode_solver.cpt_code()
                code_source_term = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", sub_pow, code_source_term)
                code_source_term = reduce(lambda x, y: x.replace(y, dictionnary_of_translation_cython[y]),
                                          dictionnary_of_translation_cython, code_source_term)
                code_source_term = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub_slices, code_source_term)
                code_source_term = re.sub(str(var_time), 'tn', code_source_term)

        if test_source_term:
            N = ode_solver.nb_of_floors
            if N>0:
                self.code += INDENT + 'cdef:\n'
                for k in range(N):
                    self.code += 2*INDENT + 'double dummy{1:1d}[{0}]\n'.format(len(indices_m), k)
            self.code += code_source_term
        if dummy_test:
            self.code += INDENT + "pass\n"
        self.code += "\n"

    def onetimestep(self, stencil, pattern):
        """
        generate the code for one time step of the Lattice Boltzmann method

        Parameters
        ----------

        stencil : :py:class:`Stencil<pyLBM.stencil.Stencil>`
          the stencil of velocities
        pattern : list
          the list of the different phases

        Returns
        -------
        code : string
          add the onetimestep function in the attribute ``code``.

        """
        # search if there is source terms in the pattern
        # HAS TO BE FIX FOR GOOD pattern
        is_source = False
        for p in pattern:
            if isinstance(p, tuple):
                if p[0] == 'source_term':
                    is_source = True
        s = ':, '*stencil.dim
        ext = s[:-2]
        #dummy1 = ["double[{0}] x".format(':'), "double[{0}] y".format(':'), "double[{0}] z".format(':')]
        dummy1 = ["double[:] x", "double[:] y", "double[:] z"]
        dummy2 = ["double x", "double y", "double z"]
        dummy1 = dummy1[:stencil.dim]
        dummy2 = dummy2[stencil.dim:]
        dummy = ", ".join(dummy1) + ", " + ", ".join(dummy2)
        self.code += """
def onetimestep(double[{0}::1] m, double[{0}::1] f, double[{0}::1] fnew, double[{2}] in_or_out, double valin, double tn, double dt, {3}):
    cdef:
        double floc[{1}]
        double mloc[{1}]
        double xloc=0., yloc=0., zloc=0.
""".format(s, stencil.nv_ptr[-1], ext, dummy)

        for i, x in enumerate(self.sorder[1:]):
            self.code += 2*INDENT + "int i{0}, n{0} = f.shape[{1}] \n".format(i, x)
        self.code += 2*INDENT + "int i, nv = f.shape[{0}]\n".format(self.sorder[0])
        self.code += 2*INDENT + "int nvsize = {0}\n".format(stencil.nv_ptr[-1])

        sind = np.argsort(self.sorder[1:])
        indent = INDENT
        for i in sind:
            self.code += indent + "for i{0} in xrange(n{0}):\n".format(i)
            indent += INDENT
            self.code += indent + "{0}loc = {0}[i{1}]\n".format(['x', 'y', 'z'][i], i)

        self.code += indent + "if in_or_out[{0}] == valin:\n".format(', '.join(['i' + str(i) for i in range(stencil.dim)]))
        indent += INDENT
        self.code += indent + "get_f(floc, f, {0})\n".format(', '.join(['i' + str(i) for i in range(stencil.dim)]))
        self.code += indent + "f2m_loc(floc, mloc)\n"
        if is_source:
            self.code += indent + "source_term(mloc, tn, {0}*dt, xloc, yloc, zloc)\n".format(0.5)
        self.code += indent + "relaxation(mloc)\n"
        if is_source:
            self.code += indent + "source_term(mloc, tn, {0}*dt, xloc, yloc, zloc)\n".format(0.5)
        self.code += indent + "m2f_loc(mloc, floc)\n"
        self.code += indent + "set_f(floc, fnew, {0})\n".format(', '.join(['i' + str(i) for i in range(stencil.dim)]))

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
        if mpi.COMM_WORLD.Get_rank() == 0:
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
            py_importer, pyx_importer = pyximport.install(build_dir=self.build_dir, inplace=True)
            self.get_module()
            pyximport.uninstall(py_importer, pyx_importer)
