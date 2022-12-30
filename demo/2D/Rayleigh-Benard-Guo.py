

"""
test: True
"""
import numpy as np
import sympy as sp
import mpi4py.MPI as mpi
import pylbm

X, Y = sp.symbols('X, Y')
rho, qx, qy, T, LA = sp.symbols('rho, qx, qy, T, LA', real=True)

def init_T(x, y, Td, Tu, xmin, xmax, ymin, ymax):
    xmid = (xmax+xmin)/2
    print((Tu-Td)*(x<1.2*xmid)*(x>0.8*xmid))
    return Td + (Tu-Td)/(ymax-ymin)*(y-ymin) + (Tu-Td)*(x<1.2*xmid)*(x>0.8*xmid)
    #return Td + (Tu-Td)/(ymax-ymin)*(y-ymin) + (Td-Tu) * (0.1*np.random.random_sample((x.shape[0],y.shape[1]))-0.05)

def bc_up(f, m, x, y, Tu):
    m[qx] = 0.
    m[qy] = 0.
    m[T] = Tu

def bc_down(f, m, x, y, Td):
    np.random.seed(1)
    m[qx] = 0.
    m[qy] = 0.
    m[T] = Td# + (Td-Tu) * 5 * (0.1*np.random.random_sample((x.shape[0],1))-0.05)

def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.mpi_topo, 'rayleigh_benard', './rayleigh_benard', im)
    h5.set_grid(x, y)
    h5.add_scalar('T', sol.m[T])
    h5.save()

def feq_NS(v, u):
    c0 = 1#LA
    x, y = sp.symbols('x, y')
    vsymb = sp.Matrix([x, y])
    w = sp.Matrix([sp.Rational(4,9)] + [sp.Rational(1, 9)]*4 + [sp.Rational(1, 36)]*4)
    f = rho*(1 + 3*u.dot(vsymb)/c0 + sp.Rational(9, 2)*u.dot(vsymb)**2/c0**2 - sp.Rational(3, 2)*u.norm()**2/c0**2)
    return sp.Matrix([w[iv]*f.subs([(x, vv[0]), (y, vv[1])]) for iv, vv in enumerate(v)])

def feq_T(v, u):
    c0 = 1#LA
    x, y = sp.symbols('x, y')
    vsymb = sp.Matrix([x, y])
    f = T/4*(1 + 2*u.dot(vsymb)/c0)
    return sp.Matrix([f.subs([(x, vv[0]), (y, vv[1])]) for iv, vv in enumerate(v)])

def run(dx, Tf, generator="cython", sorder=None, with_plot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    Tu = -0.5
    Td =  0.5
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    Ra = 100
    Pr = 0.71
    Ma = 0.01
    alpha = .005
    la = 1. # velocity of the scheme
    rhoo = 1.
    g = 9.81

    nu = np.sqrt(Pr*alpha*9.81*(Td-Tu)*(ymax-ymin)/Ra)
    diffusivity = nu/Pr
    tau = 1./(.5+3*nu)
    taup = 1./(.5+2*diffusivity)
    tau=1.99
    taup=1.991
    sf = [0]*3 + [tau]*6
    sT = [0] + [taup]*3

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1, -1, 0, 1]},
        'space_step':dx,
        'scheme_velocity':1,
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
                'feq':(feq_NS, (sp.Matrix([qx/rho, qy/rho]),)),
                'source_terms':{qy: alpha*g*T},
            },
            {
                'velocities':list(range(1, 5)),
                'conserved_moments': [T],
                'polynomials':[1, X, Y, X**2-Y**2],
                'feq':(feq_T, (sp.Matrix([qx/rho, qy/rho]),)),
                'relaxation_parameters':sT,
            },
        ],
        'init':{rho: 1.,
                qx: 0.,
                qy: 0.,
                T: (init_T, (Td, Tu, xmin, xmax, ymin, ymax))},
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack}, 'value':(bc_down, (Td,))},
            1:{'method':{0: pylbm.bc.BouzidiBounceBack, 1: pylbm.bc.BouzidiAntiBounceBack}, 'value':(bc_up, (Tu,))},
        },
        'generator': generator,
    }

    sol = pylbm.Simulation(dico)

    if with_plot:
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        image = ax.image(sol.m[T].T, cmap='cubehelix', clim=[Tu, Td+.25])

        def update(iframe):
            nrep = 1
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
    Tf = 0.6
    run(dx, Tf)
