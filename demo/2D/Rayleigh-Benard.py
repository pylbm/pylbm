
from six.moves import range
import numpy as np
import sympy as sp
import pylbm
import sys

"""

Rayleigh-Benard instability simulated by
Navier-Stokes solver D2Q9 coupled to thermic solver D2Q5

"""
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '*' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

HDF_save = False

X, Y, LA = sp.symbols('X, Y, lambda')
rho, qx, qy, T = sp.symbols('rho, qx, qy, T')

def init_T(x, y, Td, Tu, xmin, xmax, ymin, ymax):
    xmid = (xmax+xmin)/2
    ymid = (ymax+ymin)/2
    return Td + (Tu-Td)/(ymax-ymin)*(y-ymin) + Td*(x<1.1*xmid)*(x>0.9*xmid)*(y<ymid)

def bc_up(f, m, x, y, Tu):
    m[qx] = 0.
    m[qy] = 0.
    m[T] = Tu

def bc_down(f, m, x, y, Td):
    m[qx] = 0.
    m[qy] = 0.
    m[T] = Td

def bc_down_with_time(f, m, t, x, y, Td, Tu):
    m[qx] = 0.
    m[qy] = 0.
    if 0 <= t%10. <= 5:
        m[T] = Td
    else:
        m[T] = Tu

def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.mpi_topo, 'rayleigh_benard', './rayleigh_benard', im)
    h5.set_grid(x, y)
    h5.add_scalar('T', sol.m[T])
    h5.save()

def run(dx, Tf, time_bc, generator="cython", sorder=None, with_plot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    time_bc: boolean
        set the bottom boundary condition that evolves over time

    generator: pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    Td, Tu = 0.5, -0.5
    xmin, xmax, ymin, ymax = 0., 2., 0., 1.
    Ra = 2000
    Pr = 0.71
    Ma = 0.01
    alpha = .005
    la = 1. # velocity of the scheme
    rhoo = 1.
    g = 9.81

    nu = np.sqrt(Pr*alpha*9.81*(Td-Tu)*(ymax-ymin)/Ra)
    kappa = nu/Pr
    eta = nu
    snu = 1./(.5+3*nu)
    seta = 1./(.5+3*eta)
    sq = 8*(2-snu)/(8-snu)
    se = seta
    sf = [0., 0., 0., seta, se, sq, sq, snu, snu]
    a = .5
    skappa = 1./(.5+10*kappa/(4+a))
    se = 1./(.5+np.sqrt(3)/3)
    snu = se
    sT = [0., skappa, skappa, se, snu]

    if time_bc:
        bc = {
            0:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack},
               'value': (bc_down_with_time, (Td, Tu)),
               'time_bc': True},
            1:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack},
               'value':(bc_up, (Tu,))},
        }
    else:
        bc = {
            0:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack},
               'value': (bc_down, (Td,))},
            1:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack},
               'value':(bc_up, (Tu,))},
        }

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1, -1, 0, 1]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':list(range(9)),
                'conserved_moments': [rho, qx, qy],
                'polynomials':[
                    1, X, Y,
                    3*(X**2+Y**2)-4,
                    0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                    3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                    X**2-Y**2, X*Y
                ],
                'relaxation_parameters':sf,
                'equilibrium':[
                    rho, qx, qy,
                    -2*rho + 3*(qx**2+qy**2),
                    rho - 3*(qx**2+qy**2),
                    -qx, -qy,
                    qx**2 - qy**2, qx*qy
                ],
                'source_terms':{qy: alpha*g*T},
            },
            {
                'velocities':list(range(5)),
                'conserved_moments':T,
                'polynomials':[1, X, Y, 5*(X**2+Y**2) - 4, (X**2-Y**2)],
                'equilibrium':[T, T*qx, T*qy, a*T, 0.],
                'relaxation_parameters':sT,
            },
        ],
        'init':{rho: 1.,
                qx: 0.,
                qy: 0.,
                T: (init_T, (Td, Tu, xmin, xmax, ymin, ymax))},
        'boundary_conditions': bc,
        'generator': generator,
    }

    sol = pylbm.Simulation(dico)

    x, y = sol.domain.x, sol.domain.y

    if with_plot:
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        image = ax.image(sol.m[T].T, cmap='cubehelix', clim=[Tu, Td+.25])

        def update(iframe):
            nrep = 64
            for i in range(nrep):
                sol.one_time_step()
            image.set_data(sol.m[T].T)
            ax.title = "Solution t={0:f}".format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
           sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 10.
    time_bc = False
    run(dx, Tf, time_bc)
