import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import pylab as plt

import mpi4py.MPI as mpi
import pyLBM
from pyLBM.interface import get_directions

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def test1D(use_mpi):
    xmin, xmax = 0., 1.
    dx = (xmax-xmin)/128
    Tf = 0.5

    def init_u(x):
        milieu = 0.5*(xmin+xmax)
        largeur = 2*dx
        milieu -= 0.25*Tf
        return 1.0/largeur**10 * (x-milieu-largeur)**5 * (milieu-x-largeur)**5 * (abs(x-milieu)<=largeur)

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':1.,
        'schemes':[{
            'velocities':range(1, 3),
            'polynomials':Matrix([1, X]),
            'relaxation_parameters':[0., 1.5],
            'equilibrium':Matrix([u[0][0], .5*u[0][0]]),
            'init':{0:(init_u,)},
        }],
        'generator': pyLBM.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0:pyLBM.bc.bouzidi_bounce_back_1D,}, 'value':None},
        },
        'use_mpi':use_mpi,
    }

    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    if not use_mpi:
        if rank == 0:
            sol = pyLBM.Simulation(dico)
            m = sol.m[0][0][1:-1]
            f = open('test_mpi_res_npx={0:1d}.txt'.format(1), mode='w')
            f.write("# Number of processors: {0:d}\n\n".format(1))
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            while sol.t < Tf:
                sol.one_time_step()
            sol.f2m()
            f.write("\n")
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            f.close()
        else:
            m = None
    else:
        sol = pyLBM.Simulation(dico)
        if rank == 0:
            m = np.empty(sol.domain.Ng)
            comm.Gather([sol.m[0][0][1:-1].flatten(), mpi.DOUBLE],
                [m, mpi.DOUBLE], 0)
        else:
            m = None
            comm.Gather([sol.m[0][0][1:-1].flatten(), mpi.DOUBLE], None, 0)
        if rank == 0:
            f = open('test_mpi_res_npx={0:1d}.txt'.format(comm.size), mode='w')
            f.write("# Number of processors: {0:d}\n\n".format(comm.size))
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
        while sol.t < Tf:
            sol.one_time_step()
        sol.f2m()
        if rank == 0:
            comm.Gather([sol.m[0][0][1:-1].flatten(), mpi.DOUBLE],
                [m, mpi.DOUBLE], 0)
        else:
            comm.Gather([sol.m[0][0][1:-1].flatten(), mpi.DOUBLE], None, 0)
        if rank == 0:
            f.write("\n")
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            f.close()

    return m


def test2D(use_mpi):
    dx = 1./16 # spatial step
    la = 4 # velocity of the scheme
    rhoo = 1.
    deltarho = 1.
    g = 1.
    Tf = 3*dx

    Taille = 1.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    vitesse = range(1,5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    def initialization_rho(x,y):
        return rhoo * np.ones((y.shape[0], x.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

    def initialization_q(x,y):
        return np.zeros((y.shape[0], x.shape[0]), dtype='float64')

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 2., 2., 1.5],
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + 0.5*g*u[0][0]**2, u[1][0]*u[2][0]/u[0][0], 0.]),
                    'init':{0:(initialization_q,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + 0.5*g*u[0][0]**2, 0.]),
                    'init':{0:(initialization_q,)},
                    },
        ],
        'generator': pyLBM.generator.NumpyGenerator,
        'use_mpi':use_mpi,
    }

    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    nbproc = comm.Get_size()
    if not use_mpi:
        if rank == 0:
            sol = pyLBM.Simulation(dico)
            dim = sol.domain.Ng
            size = dim[0]*dim[1]
            m = np.empty((size,))
            dummy = sol.m[0][0][1:-1,1:-1]
            m[:] = dummy.flatten()
            f = open('test_mpi_res_npx={0:1d}.txt'.format(1), mode='w')
            f.write("# Number of processors: {0:d}\n\n".format(1))
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            while sol.t < Tf:
                sol.one_time_step()
            sol.f2m()
            f.write("\n")
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            f.close()
            m[:] = dummy.flatten()
        else:
            m = None
    else:
        sol = pyLBM.Simulation(dico)
        coords = comm.gather(sol.interface.get_coords(), 0)
        xmin, xmax, ymin, ymax = sol.domain.x[0][0], sol.domain.x[0][-1], sol.domain.x[1][0], sol.domain.x[1][-1],
        if rank == 0:
            dim = sol.domain.Ng
            size = dim[0]*dim[1]
            dimloc = sol.domain.N
            sizeloc = dimloc[0]*dimloc[1]
            m = np.empty((size,))
            comm.Gather([sol.m[0][0][1:-1,1:-1].flatten(), mpi.DOUBLE],
                [m, mpi.DOUBLE], 0)
        else:
            m = None
            comm.Gather([sol.m[0][0][1:-1,1:-1].flatten(), mpi.DOUBLE], None, 0)
        if rank == 0:
            f = open('test_mpi_res_npx={0:1d}.txt'.format(comm.size), mode='w')
            f.write("# Number of processors: {0:d}\n\n".format(comm.size))
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
        while sol.t < Tf:
            sol.one_time_step()
        sol.f2m()
        if rank == 0:
            comm.Gather([sol.m[0][0][1:-1,1:-1].flatten(), mpi.DOUBLE],
                [m, mpi.DOUBLE], 0)
        else:
            comm.Gather([sol.m[0][0][1:-1,1:-1].flatten(), mpi.DOUBLE], None, 0)
        if rank == 0:
            mm = np.empty(dim)
            for k in range(nbproc):
                b, a = coords[k]
                mm[a*dimloc[0]:(a+1)*dimloc[0], b*dimloc[1]:(b+1)*dimloc[1]] = m[k*sizeloc:(k+1)*sizeloc].reshape(dimloc)
            m = mm.copy()
            f.write("\n")
            np.savetxt(f, m, fmt = '%.10e', delimiter='', newline='\n')
            f.close()

    if m is None:
        return None
    else:
        return m.reshape(dim)

if __name__ == "__main__":
    # if mpi.COMM_WORLD.Get_rank() == 0:
    #     plt.figure(1)
    #     plt.clf()
    #     plt.hold(True)
    # m0 = test1D(use_mpi = False)
    # if mpi.COMM_WORLD.Get_rank() == 0:
    #     plt.plot(m0, 'r*', label='serial')
    # m1 = test1D(use_mpi = True)
    # if mpi.COMM_WORLD.Get_rank() == 0:
    #     plt.plot(m1, 'k', label='mpi')
    # if mpi.COMM_WORLD.Get_rank() == 0:
    #     plt.legend()
    #     plt.hold(False)
    #     print np.linalg.norm(m0-m1)
    #     plt.show()

    if mpi.COMM_WORLD.Get_rank() == 0:
        plt.figure(1)
        plt.clf()
        plt.figure(2)
        plt.clf()
    m0 = test2D(use_mpi = False)
    if mpi.COMM_WORLD.Get_rank() == 0:
        plt.figure(1)
        plt.imshow(m0, label='serial')
        plt.title('1 proc')
        plt.pause(1.)
    m1 = test2D(use_mpi = True)
    if mpi.COMM_WORLD.Get_rank() == 0:
        plt.figure(2)
        plt.imshow(m1, label='mpi')
        plt.title('{0:d} proc'.format(mpi.COMM_WORLD.Get_size()))
        plt.pause(1.)
    if mpi.COMM_WORLD.Get_rank() == 0:
        print np.linalg.norm(m0-m1)
        plt.show()
