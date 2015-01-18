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


def matMult(A, x, y, indent='', prefix='', suffix=''):
    """
    return a string representing the unroll matrix vector operation y = Ax

    Parameters
    ----------
    A : the matrix coefficients
    x : input vector in a string format
    y : output vector in a string format where the matrix vector product is
        stored
    indent : string representing spaces or tabs which are set at the beginning
             of each line
    prefix : string
    suffix : string

    Returns
    -------
    string representing the unroll matrix vector operation y = Ax

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
    def __init__(self, build_dir=None, suffix='.py'):
        self.build_dir = build_dir
        if build_dir is None:
            self.build_dir = tempfile.mkdtemp(suffix='LBM') + '/'
        self.f = tempfile.NamedTemporaryFile(suffix=suffix, prefix=self.build_dir + 'LBM', delete=False)
        sys.path.append(self.build_dir)
        self.code = ''

        atexit.register(self.exit)

        print self.f.name

    def setup(self):
        pass

    def f2m(self):
        pass

    def m2f(self,):
        pass

    def transport(self,):
        pass

    def compile(self):
        self.f.write(self.code)
        self.f.close()

    def get_module(self):
        return self.f.name.replace(self.build_dir, "").split('.')[0]

    def exit(self):
        print "delete generator"
        os.unlink(self.f.name)

class NumpyGenerator(Generator):
    def __init__(self, build_dir=None):
        Generator.__init__(self, build_dir)

        sys.path.append(self.build_dir)

    def transport(self, ns, stencil, dtype = 'f8'):
        self.code += "def transport(f):\n"
        ind = 0
        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                s1 = ''
                s2 = ''
                toInput = False
                v = stencil.v[k][i].v
                print v
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

    def m2f(self, A, k, dim):
        self.code += "def m2f_{0:d}(m, f):\n".format(k)
        self.code += matMult(A, 'm', 'f', '\t')
        self.code += "\n"

    def f2m(self, A, k, dim, dtype = 'f8'):
        self.code += "def f2m_{0:d}(f, m):\n".format(k)
        self.code += matMult(A, 'f', 'm', '\t')
        self.code += "\n"


class NumbaGenerator(Generator):
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
    def __init__(self, build_dir=None):
        Generator.__init__(self, build_dir, suffix='.pyx')

    def setup(self):
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
        Generator.compile(self)
        bld = open(self.f.name.replace('.pyx', '.pyxbld'), "w")
        code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     extra_compile_args = ['-O3', '-w'])
        """
#         code = """
# def make_ext(modname, pyxfilename):
#     from distutils.extension import Extension
#
#     return Extension(name = modname,
#                      sources=[pyxfilename],
#                      extra_compile_args = ['-O3', '-fopenmp', '-w'],
#                      extra_link_args= ['-fopenmp'])
#         """
        bld.write(code)
        bld.close()

        import pyximport
        pyximport.install(build_dir= self.build_dir, inplace=True)

class CythonGeneratorOld(Generator):
    def __init__(self, build_dir=None):
        Generator.__init__(self, build_dir, suffix='.pyx')

    def setup(self):
        self.code += """
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#import cython
#from cython.parallel import prange

"""

    def m2f(self, A, k, dim, dtype = 'double'):
        ext = ''
        for i in xrange(dim):
            ext += ', :'

        self.code += "def m2f_{0:d}({1}[:, ::1] m, {1}[:, ::1] f):\n".format(k, dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = f.shape[1] \n"

        #self.code += "\tfor i in prange(n, nogil=True, schedule='dynamic'):\n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t\t"
        ext = ', i'

        self.code += matMult(A, 'm', 'f', tab, suffix=ext)
        self.code += "\n"

    def f2m(self, A, k, dim, dtype = 'double'):
        ext = ''
        for i in xrange(dim):
            ext += ', :'

        self.code += "def f2m_{0:d}({1}[:, ::1] f, {1}[:, ::1] m):\n".format(k, dtype)
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[1] \n"
        #self.code += "\tfor i in prange(n, nogil=True, schedule='dynamic'):\n"
        self.code += "\tfor i in xrange(n):\n"

        tab = "\t"*2
        ext = ', i'
        self.code += matMult(A, 'f', 'm', tab, suffix=ext)
        self.code += "\n"

    def transport(self, ns, stencil, dtype = 'double'):
        self.code += "def transport({0}[:{1}:1] f):\n".format(dtype, ', :'*stencil.dim)
        self.code += "\tcdef:\n"

        for i in xrange(stencil.dim):
            self.code += "\t\tint i{0}, n{0} = f.shape[{1:d}] \n".format(i, i+1)

        ind = 0
        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                s1 = ''
                s2 = ''
                tab = '\t'
                loop = ''
                toInput = False
                v = stencil.v[k][i].v
                lv = len(v)
                for iv in xrange(len(v)-1, -1, -1):
                    s1 += ', i{0:d}'.format(lv - 1 - iv)

                    if v[iv] > 0:
                        toInput = True
                        loop += tab + 'for i{0:d} in xrange(n{0:d} - 1, {1:d} - 1, -1):\n'.format(lv - 1 - iv, v[iv])
                        s2 += ', i{0:d} - {1:d}'.format(lv - 1 - iv, v[iv])
                        tab += '\t'
                    elif v[iv] < 0:
                        toInput = True
                        loop += tab + 'for i{0:d} in xrange(0, n{0:d} - {1:d}):\n'.format(lv - 1 - iv, -v[iv])
                        s2 += ', i{0:d} + {1:d}'.format(lv - 1 - iv, -v[iv])
                        tab += '\t'
                    else:
                        loop += tab + 'for i{0:d} in xrange(0, n{0:d}):\n'.format(lv - 1 - iv)
                        s2 += ', i{0:d}'.format(lv - 1 - iv)
                        tab += '\t'

                if toInput:
                    self.code += loop + tab + "\tf[{0:d}{1}] = f[{0:d}{2}]\n".format(ind, s1, s2)
                ind += 1
        self.code += "\n"

    def equilibrium(self, ns, stencil, eq, la, dtype = 'f8'):
        self.code += "def equilibrium(double [:, ::1] m):\n"
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, n = m.shape[1] \n"
        self.code += "\tfor i in xrange(n):\n"

        def subpow(g):
            s = g.group('m')
            for i in xrange(int(g.group('pow')) - 1):
                s += '*' + g.group('m')
            return s

        def sub(g):
            i = int(g.group('i'))
            j = int(g.group('j'))

            return '[' + str(stencil.nv_ptr[i] + j) + ', i]'


        for k in xrange(ns):
            for i in xrange(stencil.nv[k]):
                if str(eq[k][i]) != "m[%d][%d]"%(k,i):
                    if eq[k][i] != 0:
                        str2input = str(eq[k][i].subs([(sp.symbols('LA'), la),]))
                        res = re.sub("(?P<m>\w*\[\d\]\[\d\])\*\*(?P<pow>\d)", subpow, str2input)
                        res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub, res)

                        #res = re.sub("\[(?P<i>\d)\]\[(?P<j>\d)\]", sub,
                        #           str(eq[k][i].subs([(sp.symbols('LA'), la),])))
                        self.code += "\t\tm[%d, i] = %s\n"%(stencil.nv_ptr[k] + i, res)
                    else:
                        self.code += "\t\tm[%d, i] = 0.\n"%(stencil.nv_ptr[k] + i)
        self.code += "\n"

    def relaxation(self, ns, stencil, s, eq, la, dtype = 'f8'):
        self.code += "def relaxation(double [:, ::1] m):\n"
        self.code += "\tcdef:\n"

        self.code += "\t\tint i, j, n = m.shape[1] \n"
        self.code += "\t\tdouble temp[%d] \n"%stencil.nv_ptr[-1]
        self.code += "\tfor i in xrange(n):\n"
        self.code += "\t\tfor j in xrange(%d):\n"%stencil.nv_ptr[-1]
        self.code += "\t\t\ttemp[j] = m[j, i]\n"

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
                        res = res.replace('m[', 'temp[')

                        self.code += "\t\ttemp[{0:d}] += {1:.16f}*({2} - temp[{0:d}])\n".format(stencil.nv_ptr[k] + i, s[k][i], res)
                    else:
                        self.code += "\t\ttemp[{0:d}] *= (1. - {1:.16f})\n".format(stencil.nv_ptr[k] + i, s[k][i])
        self.code += "\t\tfor j in xrange(%d):\n"%stencil.nv_ptr[-1]
        self.code += "\t\t\tm[j, i] = temp[j]\n"
        self.code += "\n"

    def compile(self):
        Generator.compile(self)
        #bld = open(self.f.name.replace('.pyx', '.pyxbld'), "w")
        code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     #extra_compile_args = ['-O3', '-fopenmp'],
                     #extra_link_args= ['-fopenmp'])

"""
        #bld.write(code)
        #bld.close()

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
