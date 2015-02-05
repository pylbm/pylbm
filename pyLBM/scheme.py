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

from .stencil import Stencil
from .generator import *

from .logs import __setLogger
log = __setLogger(__name__)


X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

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
      - generator : a generator for the code, optional
        (see :py:class:`Generator <pyLBM.generator.Generator>`)

    Notes
    -----

    Each dictionary of the list `schemes` should contains the following `key:value`

    - velocities : list of the velocities number
    - polynomials : sympy matrix of the polynomial functions that define the moments
    - equilibrium : sympy matrix of the values that define the equilibrium
    - relaxation_parameters : list of the value of the relaxation parameters
    - init : a dictionary to define the initial conditions (see examples)

    If the stencil has already been computed, it can be pass in argument.

    Attributes
    ----------

    dim : int
      spatial dimension
    la : double
      scheme velocity, ratio dx/dt
    nscheme : int
      number of elementary schemes
    stencil : object of class :py:class:`Stencil <pyLBM.stencil.Stencil>`
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
    generator : :py:class:`Generator <pyLBM.generator.Generator>`
      the used generator (
      :py:class:`NumpyGenerator<pyLBM.generator.NumpyGenerator>`,
      :py:class:`CythonGenerator<pyLBM.generator.CythonGenerator>`,
      ...)

    Methods
    -------

    create_moments_matrix :
      Create the moments matrices
    create_relaxation_function :
      Create the relaxation function
    create_equilibrium_function :
      Create the equilibrium function
    create_transport_function :
      Create the transport function
    create_f2m_function :
      Create the function f2m
    create_m2f_function :
      Create the function m2f

    generate :
      Generate the code

    equilibrium :
      Compute the equilibrium
    transport :
      Transport phase
    relaxation :
      Relaxation phase
    f2m :
      Compute the moments from the distribution functions
    m2f :
      Compute the distribution functions from the moments
    onetimestep :
      One time step of the Lattice Boltzmann method
    set_boundary_conditions :
      Apply the boundary conditions

    Examples
    --------

    see demo/examples/scheme/

    """
    def __init__(self, dico, stencil=None):
        if stencil is not None:
            self.stencil = stencil
        else:
            self.stencil = Stencil(dico)
        self.dim = self.stencil.dim
        self.la = dico['scheme_velocity']
        self.nscheme = self.stencil.nstencils
        scheme = dico['schemes']
        self.P = [s['polynomials'] for s in scheme]
        self.EQ = [s['equilibrium'] for s in scheme]
        self.s = [s['relaxation_parameters'] for s in scheme]
        self.create_moments_matrices()

        #self.nv_on_beg = nv_on_beg
        self.generator = dico.get('generator', NumpyGenerator)()
        log.info("Generator used for the scheme functions:\n{0}\n".format(self.generator))
        #print self.generator
        if isinstance(self.generator, CythonGenerator):
            self.nv_on_beg = False
        else:
            self.nv_on_beg = True
        log.debug("Message from scheme.py: nv_on_beg = {0}".format(self.nv_on_beg))
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
                log.error('Function create_moments_matrices: the dimension is not correct')
            try:
                self.invM.append(self.M[-1].inv())
            except:
                s = 'Function create_moments_matrices: M is not invertible\n'
                s += 'The choice of polynomials is odd in the elementary scheme number {0:d}'.format(compt)
                log.error(s)
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
        """
        Generate the code by using the appropriated generator

        Notes
        -----

        The code can be viewed. If S is the scheme

        >>> print S.generator.code
        """
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
        """ Compute the distribution functions f from the moments m """
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
        """ Compute the moments m from the distribution functions f """
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
        """ The transport phase on the distribution functions f """
        exec "from %s import *"%self.generator.get_module()
        exec "transport(f)"

    def equilibrium(self, m):
        """ Compute the equilibrium """
        exec "from %s import *"%self.generator.get_module()
        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])
            exec "equilibrium(m.reshape(({0}, {1})))".format(self.stencil.nv_ptr[-1], space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "equilibrium(m.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])

    def relaxation(self, m):
        """ The relaxation phase on the moments m """
        exec "from %s import *"%self.generator.get_module()
        if self.nv_on_beg:
            space_size = np.prod(m.shape[1:])
            exec "relaxation(m.reshape(({0}, {1})))".format(self.stencil.nv_ptr[-1], space_size)
        else:
            space_size = np.prod(m.shape[:-1])
            exec "relaxation(m.reshape(({0}, {1})))".format(space_size, self.stencil.nv_ptr[-1])

    def onetimestep(self, m, f, fcuurent, in_or_out, valin):
        """ Compute one time step of the Lattice Boltzmann method """
        exec "from %s import *"%self.generator.get_module()
        exec "onetimestep(m, f, fcuurent, in_or_out, valin)"

    def set_boundary_conditions(self, f, m, bc, nv_on_beg):
        """
        Compute the boundary conditions

        Parameters
        ----------

        f : numpy array
          the array of the distribution functions
        m : numpy array
          the array of the moments
        bc : :py:class:`pyLBM.boundary.Boundary`
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
                                if nv_on_beg:
                                    bc.floc[l][n][k] = floc[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]]
                                else:
                                    bc.floc[l][n][k] = floc[:, self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]]
                            else:
                                bc.floc[l][n][k] = None

                            # ####### TEST for Poiseuille flow
                            # if floc is not None:
                            #     print "#"*50
                            #     print "label: {0}, scheme: {1}, velocity: {2}".format(l, n, k)
                            #     if (n == 0):
                            #         print floc[4*n, :] - floc[4*n+2, :]
                            #     else:
                            #         print floc[4*n, :] + floc[4*n+2, :]
                            # ####### FIN TEST
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
