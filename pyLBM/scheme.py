# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys

import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import stencil as pyLBMSten
import generator as pyLBMGen

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

class Scheme:
    """
    Scheme class

    * Arguments

        - dim:        spatial dimension
        - Stencil:    a stencil of velocities (object of class :py:class:`LBMpy.Stencil.Stencil`)
        - P:          list of polynomials that define the moments
        - EQ:         list of the equilibrium functions
        - s:          list of relaxation parameters
        - la:         scheme velocity

    * Attributs

        - dim:        spatial dimension
        - la:         scheme velocity
        - nscheme:    number of elementary schemes
        - Stencil:    a stencil of velocities (object of class :py:class:`LBMpy.Stencil.Stencil`)
        - P:          list of polynomials that define the moments
        - EQ:         list of the equilibrium functions
        - s:          list of relaxation parameters
        - M:          the symbolic matrix of the moments
        - Mnum:       the numeric matrix of the moments (m = Mnum F)
        - invM:       the symbolic inverse matrix
        - invMnum:    the numeric inverse matrix (F = invMnum m)

        - Code_Transport:     Code of the function Transport
        - Code_Equilibrium:   Code of the function Equilibrium
        - Code_Relaxation:    Code of the function Relaxation
        - Code_m2F:           Code of the function m2F
        - Code_F2m:           Code of the function F2m


    * Members

        - __str__:    Function used to print informations of the scheme
        - create_moments_matrix: Function that creates the moments matrices
        - create_relaxation_function: Function that creates the relaxation function
        - create_equilibrium_function: Function that creates the equilibrium function
        - create_transport_function: Function that creates the transport function
        - create_f2m_function:Function that creates the function f2m
        - create_m2f_function:Function that creates the function m2f

        All the created functions relaxation, equilibrium, transport, f2m, and m2f
        are member functions of the class :py:class:`LBMpy.Scheme.Scheme` and have a unique argument of type
        :py:class:`LBMpy.Solution.Solution`

    """
    def __init__(self, dico, stencil=None, nv_on_beg=True):
        if stencil is not None:
            self.stencil = stencil
        else:
            self.stencil = pyLBMSten.Stencil(dico)
        self.dim = self.stencil.dim
        self.la = dico['scheme_velocity']
        self.nscheme = self.stencil.nstencils
        self.P = [dico[k]['polynomials'] for k in xrange(self.nscheme)]
        self.EQ = [dico[k]['equilibrium'] for k in xrange(self.nscheme)]
        self.s = [dico[k]['relaxation_parameters'] for k in xrange(self.nscheme)]
        self.create_moments_matrices()

        #self.nv_on_beg = nv_on_beg
        self.generator = dico.get('generator', pyLBMGen.NumpyGenerator)()
        print self.generator
        if isinstance(self.generator,pyLBMGen.CythonGenerator):
            self.nv_on_beg = False
        else:
            self.nv_on_beg = True
        print "*"*50
        print "Message from scheme.py: nv_on_beg = {0}".format(self.nv_on_beg)
        print "*"*50
        self.generate()

        self.bc_compute = True

    def __str__(self):
        s = "Scheme informations\n"
        s += "\t spatial dimension: dim={0:d}\n".format(self.dim)
        s += "\t number of schemes: nscheme={0:d}\n".format(self.nscheme)
        s += "\t number of velocities:\n"
        for k in xrange(self.nscheme):
            s += "    Stencil.nv[{0:d}]=".format(k) + str(self.stencil.nv[k]) + "\n"
        s += "\t velocities value:\n"
        for k in xrange(self.nscheme):
            s+="    v[{0:d}]=".format(k)
            for v in self.stencil.v[k]:
                s += v.__str__() + ', '
            s += '\n'
        s += "\t polynomials:\n"
        for k in xrange(self.nscheme):
            s += "    P[{0:d}]=".format(k) + self.P[k].__str__() + "\n"
        s += "\t equilibria:\n"
        for k in xrange(self.nscheme):
            s += "    EQ[{0:d}]=".format(k) + self.EQ[k].__str__() + "\n"
        s += "\t relaxation parameters:\n"
        for k in xrange(self.nscheme):
            s += "    s[{0:d}]=".format(k) + self.s[k].__str__() + "\n"
        s += "\t moments matrices\n"
        s += "M = " + self.M.__str__() + "\n"
        s += "invM = " + self.invM.__str__() + "\n"
        return s

    def create_moments_matrices(self):
        """
        Create the moments matrices M and M^{-1} used to transform the repartition functions into the moments

        Example

        >>> create_moments_matrices()
        """
        self.M, self.invM = [], []
        self.Mnum, self.invMnum = [], []
        compt=0
        for v, p in zip(self.stencil.v, self.P):
            compt+=1
            lv = len(v)
            self.M.append(zeros(lv, lv))
            if self.dim == 1:
                for i in xrange(lv):
                    for j in xrange(lv):
                        self.M[-1][i, j] = p[i].subs([(X, v[j].vx),])
            elif self.dim == 2:
                for i in xrange(lv):
                    for j in xrange(lv):
                        self.M[-1][i, j] = p[i].subs([(X, v[j].vx), (Y, v[j].vy)])
            elif self.dim == 3:
                for i in xrange(lv):
                    for j in xrange(lv):
                        self.M[-1][i, j] = p[i].subs([(X, v[j].vx), (Y, v[j].vy), (Z, v[j].vz)])
            else:
                print 'Function create_moments_matrices: the dimension is not correct'
            try:
                self.invM.append(self.M[-1].inv())
            except:
                print 'Function create_moments_matrices: M is not invertible'
                print 'The choice of polynomials is odd in the elementary scheme number {0:d}'.format(compt)
                sys.exit()

        self.MnumGlob = np.zeros((self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1]))
        self.invMnumGlob = np.zeros((self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1]))

        for k in xrange(self.nscheme):
            nvk = self.stencil.nv[k]
            self.Mnum.append(np.empty((nvk, nvk), dtype='float64'))
            self.invMnum.append(np.empty((nvk, nvk), dtype='float64'))
            for i in xrange(nvk):
                for j in xrange(nvk):
                    self.Mnum[k][i, j] = (float)(self.M[k][i, j].subs([(LA,self.la),]))
                    self.invMnum[k][i, j] = (float)(self.invM[k][i, j].subs([(LA,self.la),]))
                    self.MnumGlob[self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j] = self.Mnum[k][i, j]
                    self.invMnumGlob[self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j] = self.invMnum[k][i, j]

    def generate(self):
        self.generator.setup()

        if self.nv_on_beg:
            for k in xrange(self.nscheme):
                self.generator.m2f(self.invMnum[k], k, self.dim)
                self.generator.f2m(self.Mnum[k], k, self.dim)
        else:
            # ne marche que pour cython
            self.generator.m2f(self.invMnumGlob, 0, self.dim)
            self.generator.f2m(self.MnumGlob, 0, self.dim)
            self.generator.onetimestep(self.stencil)

        self.generator.transport(self.nscheme, self.stencil)
        self.generator.equilibrium(self.nscheme, self.stencil, self.EQ, self.la)
        self.generator.relaxation(self.nscheme, self.stencil, self.s, self.EQ, self.la)
        self.generator.compile()

    def m2f(self, m, f):
        exec "from %s import *"%self.generator.get_module()
        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])
            for k in xrange(self.nscheme):
                s = slice(self.stencil.nv_ptr[k], self.stencil.nv_ptr[k + 1])
                nv = self.stencil.nv[k]
                exec "m2f_{0}(m[{1}].reshape(({2}, {3})), f[{1}].reshape(({2}, {3})))".format(k, s, nv, space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "m2f(m.reshape(({0}, {1})), f.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])

    def f2m(self, f, m):
        exec "from %s import *"%self.generator.get_module()

        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])

            for k in xrange(self.nscheme):
                s = slice(self.stencil.nv_ptr[k], self.stencil.nv_ptr[k + 1])
                nv = self.stencil.nv[k]
                exec "f2m_{0}(f[{1}].reshape(({2}, {3})), m[{1}].reshape(({2}, {3})))".format(k, s, nv, space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "f2m(f.reshape(({0}, {1})), m.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])


    def transport(self, f):
        exec "from %s import *"%self.generator.get_module()
        exec "transport(f)"

    def equilibrium(self, m):
        exec "from %s import *"%self.generator.get_module()

        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])
            exec "equilibrium(m.reshape(({0}, {1})))".format(self.stencil.nv_ptr[-1], space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "equilibrium(m.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])

    def relaxation(self, m):
        exec "from %s import *"%self.generator.get_module()

        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])
            exec "relaxation(m.reshape(({0}, {1})))".format(self.stencil.nv_ptr[-1], space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "relaxation(m.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])

    def onetimestep(self, m, f, fcuurent, in_or_out, valin):
        exec "from %s import *"%self.generator.get_module()
        exec "onetimestep(m, f, fcuurent, in_or_out, valin)"

    def set_boundary_conditions(self, f, m, bc, nv_on_beg):
        # periodic conditions
        ###################################################################################
        #                 |   0     ...        N-1    |
        # 0 1 ... lbord-1 | lbord   ...     N+lbord-1 | N+lbord ... Na-1 = N+2lbord-1
        #
        ###################################################################################
        if nv_on_beg:
            Na = f.shape[1:]
            lbord = [self.stencil.vmax[k] for k in xrange(self.dim)]
            N  = [Na[k] - 2*lbord[k] for k in xrange(self.dim)]

            for n in xrange(self.nscheme): # loop over the stencils
                s = slice(self.stencil.nv_ptr[n], self.stencil.nv_ptr[n + 1])
                f[s,:lbord[0],:]           = f[s,N[0]:N[0]+lbord[0],:]  # east
                f[s,:,:lbord[1]]           = f[s,:,N[1]:N[1]+lbord[1]]  # south
                f[s,N[0]+lbord[0]:Na[0],:] = f[s,lbord[0]:2*lbord[0],:] # west
                f[s,:,N[1]+lbord[1]:Na[1]] = f[s,:,lbord[1]:2*lbord[1]] # north
                f[s,:lbord[0],:lbord[1]]   = f[s,N[0]:N[0]+lbord[0],N[1]:N[1]+lbord[1]]  # east-south
                f[s,:lbord[0],N[1]+lbord[1]:Na[1]]   = f[s,N[0]:N[0]+lbord[0],lbord[1]:2*lbord[1]]  # east-north
                f[s,N[0]+lbord[0]:Na[0],:lbord[1]]   = f[s,lbord[0]:2*lbord[0],N[1]:N[1]+lbord[1]]  # west-south
                f[s,N[0]+lbord[0]:Na[0],N[1]+lbord[1]:Na[1]]   = f[s,lbord[0]:2*lbord[0],lbord[1]:2*lbord[1]]  # west-north
        else:
            Na = f.shape[:-1]
            lbord = [self.stencil.vmax[k] for k in xrange(self.dim)]
            N  = [Na[k] - 2*lbord[k] for k in xrange(self.dim)]
            for n in xrange(self.nscheme): # loop over the stencils
                s = slice(self.stencil.nv_ptr[n], self.stencil.nv_ptr[n + 1])
                f[:lbord[0], :, s] = f[N[0]:N[0] + lbord[0] , :, s]  # east
                f[:, :lbord[1], s] = f[:, N[1]:N[1] + lbord[1], s]  # south
                f[N[0] + lbord[0]:Na[0], :, s] = f[lbord[0]:2*lbord[0], :, s] # west
                f[:, N[1] + lbord[1]:Na[1], s] = f[:, lbord[1]:2*lbord[1], s] # north
                f[:lbord[0], :lbord[1], s] = f[N[0]:N[0] + lbord[0], N[1]:N[1] + lbord[1], s]  # east-south
                f[:lbord[0], N[1] + lbord[1]:Na[1], s] = f[N[0]:N[0] + lbord[0], lbord[1]:2*lbord[1], s]  # east-north
                f[N[0] + lbord[0]:Na[0], :lbord[1], s] = f[lbord[0]:2*lbord[0], N[1]:N[1] + lbord[1], s]  # west-south
                f[N[0] + lbord[0]:Na[0],N[1] + lbord[1]:Na[1], s] = f[lbord[0]:2*lbord[0], lbord[1]:2*lbord[1], s]  # west-north

        # non periodic conditions
        if self.bc_compute:
            #bc.floc = np.array([None]*len(bc.be))
            bc.floc = [[[None for  k in xrange(self.stencil[ind_scheme].nv)] for ind_scheme in xrange(self.nscheme)] for l in xrange(len(bc.be))]

        import copy
        for l in xrange(len(bc.be)): # loop over the labels
            for n in xrange(self.nscheme): # loop over the stencils
                for k in xrange(self.stencil[n].nv): # loop over the velocities
                    bv = bc.be[l][n][k]
                    #assert k == self.stencil.unum2index[bv.v.num]

                    if bv.distance.size != 0:
                        iy = bv.indices[0]
                        ix = bv.indices[1]
                        s  = bv.distance
                        if self.bc_compute:
                            if (bc.value_bc[l] is not None):
                                if nv_on_beg:
                                    mloc = np.ascontiguousarray(m[:, iy, ix])
                                else:
                                    mloc = np.ascontiguousarray(m[iy, ix, :])
                                floc = np.zeros(mloc.shape)
                                bc.value_bc[l](floc, mloc,
                                               bc.domain.x[0][ix] + s*bv.v.vx*bc.domain.dx,
                                               bc.domain.x[1][iy] + s*bv.v.vy*bc.domain.dx, self)
                            else:
                                floc = None

                            bc.floc[l][n][k] = floc
                            #bc.floc[l][n][k] = None
                        if nv_on_beg:
                            bc.method_bc[l][n](f[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]], bv, self.stencil[n].num2index, bc.floc[l][n][k], nv_on_beg)
                        else:
                            bc.method_bc[l][n](f[:, :, self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]], bv, self.stencil[n].num2index, bc.floc[l][n][k], nv_on_beg)
                            #bc.method_bc[l][n](f, bv, self.stencil[n].num2index, bc.floc[l][n][k], nv_on_beg)
        self.bc_compute = False


def test_1D(opt):
    dim = 1 # spatial dimension
    la = 1.
    print "\n\nTest number {0:d} in {1:d}D:".format(opt,dim)
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
        dico[0] = {'velocities':range(5),
           'polynomials':Matrix([1, la*X, X**2/2, X**3/2, X**4/2]),
           'equilibrium':Matrix([u[0][0], u[0][1], (0.5*la)**2/2*u[0][0], 0, 0]),
           'relaxation_parameters':[0,0,1.9, 1., 1.]
           }
    try:
        LBMscheme = Scheme(dico)
        print LBMscheme
        return 1
    except:
        return 0

def test_2D(opt):
    dim = 2 # spatial dimension
    la = 1.
    print "\n\nTest number {0:d} in {1:d}D:".format(opt,dim)
    dico = {'dim':dim, 'scheme_velocity':la}
    if (opt == 0):
        dico['number_of_schemes'] = 2 # number of elementary schemes
        dico[0] = {'velocities':range(1,5),
           'polynomials':Matrix([1, la*X, la*Y, X**2-Y**2]),
           'equilibrium':Matrix([u[0][0], .1*u[0][0], 0, 0]),
           'relaxation_parameters':[0, 1, 1, 1]
           }
        dico[1] = {'velocities':range(5),
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
        dico[0] = {'velocities':range(9),
           'polynomials':Matrix([1, la*X, la*Y, 3*(X**2+Y**2)-4, (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2, 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y, X**2-Y**2, X*Y]),
           'equilibrium':Matrix([u[0][0], u[0][1], u[0][3], -2*u[0][0] + 3*q2, u[0][0]+1.5*q2, u[0][1]/la, u[0][2]/la, qx2-qy2, qxy]),
           'relaxation_parameters':[0, 0, 0, 1, 1, 1, 1, 1, 1]
           }
    try:
        LBMscheme = Scheme(dico)
        print LBMscheme
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
